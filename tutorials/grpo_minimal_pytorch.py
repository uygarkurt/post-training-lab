"""Minimal GRPO training in PyTorch on an NVIDIA GPU."""

import copy
import os
import random
import re

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2-0.5B-Instruct"
GROUP_SIZE = 4
MAX_NEW_TOKENS = 256
LEARNING_RATE = 1e-6
KL_COEFFICIENT = 0.02
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
NUM_ITERATIONS = 500
EPSILON = 1e-8

LORA_RANK = 8
LORA_ALPHA = 16.0

SEED = 42
VALIDATION_SPLIT = 0.05
MAX_PROMPT_LENGTH = 512
EVALUATE_EVERY = 50
EVALUATION_BATCH_SIZE = 8

SAVE_EVERY = 100
CHECKPOINT_DIRECTORY = "./checkpoints/tutorials/grpo_minimal_pytorch"

DATASET_NAME = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT = "train"

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:)\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _load_rows(split=DATASET_SPLIT):
    return hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=split)


def set_random_seed(seed):
    """Seed all RNGs and require deterministic CUDA algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_policy_and_reference():
    """Load the initial GRPO policy and an identical frozen reference model."""
    print(f"Loading model {MODEL} ...")
    policy = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA training without an adapter attaches a new adapter to the full model.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules="all-linear",
    )
    print("Applying new LoRA adapters to all linear layers ...")
    policy = get_peft_model(policy, lora_config)

    # GRPO's reference must represent the policy before its first update.
    # Keep dropout disabled for consistent log-probability comparisons; eval mode still allows gradients.
    policy.eval()
    reference = copy.deepcopy(policy)
    reference.requires_grad_(False)
    reference.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    print(
        f"Training mode: lora | "
        f"{trainable_parameters:,} / {total_parameters:,} parameters trainable"
    )

    return policy, reference, tokenizer


def _normalize_number(raw):
    """Remove whitespace, thousands separators, and a trailing decimal point."""
    value = raw.strip().replace(",", "")
    return value.removesuffix(".")


def extract_gsm8k_answer(text):
    """Parse the final numeric answer after the GSM8K #### marker."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return _normalize_number(match.group(1))


def extract_final_answer(text):
    """Extract the most likely final numeric answer from a model completion."""
    if not text or not text.strip():
        return None

    match = _ANSWER_RE.search(text)
    if match:
        return _normalize_number(match.group(1))

    match = _ANSWER_IS_RE.search(text)
    if match:
        return _normalize_number(match.group(1))

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def answers_match(prediction, ground_truth):
    """Compare two answers numerically when possible."""
    if prediction is None:
        return False

    try:
        return abs(float(prediction) - float(ground_truth)) < 1e-5
    except ValueError:
        return prediction == ground_truth


def is_answer_correct(completion_text, ground_truth):
    """Return whether a model completion contains the expected final answer."""
    predicted_answer = extract_final_answer(completion_text)
    return answers_match(predicted_answer, ground_truth)


def answer_rewards(rollouts_text, ground_truth):
    """Return binary rewards for decoded model completions."""
    rewards = []

    for text in rollouts_text:
        is_correct = is_answer_correct(text, ground_truth)
        rewards.append(1.0 if is_correct else 0.0)

    return rewards


def _make_grpo_collate_fn(pad_id):
    """Build a collator that left-pads GRPO prompts for generation."""
    def collate_fn(batch):
        max_prompt_length = max(len(sample["prompt_ids"]) for sample in batch)
        prompt_ids = []
        prompt_attention_masks = []

        for sample in batch:
            padding_length = max_prompt_length - len(sample["prompt_ids"])
            prompt_ids.append(
                [pad_id] * padding_length + sample["prompt_ids"]
            )
            prompt_attention_masks.append(
                [0] * padding_length + [1] * len(sample["prompt_ids"])
            )

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(
                prompt_attention_masks,
                dtype=torch.long,
            ),
            "ground_truth": [sample["ground_truth"] for sample in batch],
        }

    return collate_fn


def build_grpo_dataloader(dataset, pad_id, batch_size):
    """Build a deterministic DataLoader of padded GRPO prompts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_grpo_collate_fn(pad_id),
        num_workers=0,
        drop_last=False,
    )


class GSM8KGRPODataset(Dataset):
    """Tokenize and hold GSM8K prompts and answers for GRPO."""

    def __init__(self, tokenizer, max_prompt_len, split=DATASET_SPLIT):
        self.samples = []
        self.skipped = 0

        for row in _load_rows(split):
            question = row["question"]
            answer = row["answer"]

            ground_truth = extract_gsm8k_answer(answer)
            if ground_truth is None:
                self.skipped += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt_text)

            if len(prompt_ids) >= max_prompt_len:
                self.skipped += 1
                continue

            self.samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "ground_truth": ground_truth,
                    "question": question,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @classmethod
    def build_train_val_datasets(
        cls,
        tokenizer,
        max_prompt_len,
        val_split,
        seed,
    ):
        """Build GRPO train and validation datasets."""
        dataset = cls(tokenizer, max_prompt_len=max_prompt_len)

        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        print(
            f"  {len(dataset)} samples loaded, {dataset.skipped} skipped  "
            f"→  {len(train_dataset)} train / {len(val_dataset)} val  "
            f"({val_split * 100:.0f}% val split)."
        )
        return train_dataset, val_dataset


def validate(
    policy,
    val_dataset,
    tokenizer,
    max_new_tokens,
    batch_size,
    description="val",
):
    """Calculate accuracy by greedily generating over a batched dataset."""
    correct_answers = 0
    total_answers = 0
    device = next(policy.parameters()).device
    val_loader = build_grpo_dataloader(
        val_dataset,
        pad_id=tokenizer.pad_token_id,
        batch_size=batch_size,
    )

    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc=f"  {description}",
            leave=False,
            unit="batch",
        ):
            prompt_tensor = batch["prompt_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)

            generated_ids = policy.generate(
                input_ids=prompt_tensor,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                use_cache=True,
            )

            prompt_length = prompt_tensor.shape[1]
            completion_ids = generated_ids[:, prompt_length:]
            completion_texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )

            correct_answers += sum(
                is_answer_correct(text, ground_truth)
                for text, ground_truth in zip(
                    completion_texts,
                    batch["ground_truth"],
                )
            )
            total_answers += len(batch["ground_truth"])

    if total_answers == 0:
        return float("nan")
    return correct_answers / total_answers


def generate_rollouts(policy, prompt_tensor, group_size, max_new_tok, tokenizer):
    # All allowing mask. Removes EOS and PAD token ambiguity.
    prompt_attention_mask = torch.ones_like(prompt_tensor)

    with torch.no_grad():
        generated_ids = policy.generate(
            input_ids=prompt_tensor,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tok,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
        )  # [G, P + L]

    prompt_length = prompt_tensor.shape[1]  # P
    completion_ids = generated_ids[:, prompt_length:]  # [G, L]

    # Works for Qwen. It produces EOS after generation ends. Fills the rest with PAD.
    is_eos_or_pad = (completion_ids.eq(tokenizer.eos_token_id) | completion_ids.eq(tokenizer.pad_token_id))

    # We have to include the first EOS token
    completion_mask = (is_eos_or_pad.cumsum(dim=-1) <= 1).long()  # [G, L]
    truncated_rollouts = ~is_eos_or_pad.any(dim=-1)  # [G]

    return completion_ids, completion_mask, truncated_rollouts


def token_logprobs(model, prompt_ids, rollouts, rollout_masks):
    prompt_ids_extended = prompt_ids.repeat_interleave(rollouts.shape[0], dim=0)  # [G, P]
    complete_ids = torch.cat([prompt_ids_extended, rollouts], dim=-1)  # [G, P + L]

    logits = model(complete_ids, use_cache=False).logits  # [G, P + L, V]

    prompt_length = prompt_ids.shape[1]

    # Get end of the prompt token. Since it predicts the first generated token. Don't take last token. It's out of scope.
    logits_shift = logits[:, prompt_length - 1:-1, :]  # [G, L, V]

    logprobs = torch.log_softmax(logits_shift, dim=-1)  # [G, L, V]

    # Choose probabilities that corresponds to tokens in the rollout with fancy indexing.
    group_indices = torch.arange(rollouts.shape[0], device=rollouts.device).unsqueeze(-1)
    position_indices = torch.arange(rollouts.shape[1], device=rollouts.device).unsqueeze(0)
    selected_logprobs = logprobs[group_indices, position_indices, rollouts]  # [G, L]

    # Apply mask
    selected_logprobs_masked = (
        selected_logprobs * rollout_masks
    )

    return selected_logprobs_masked


def main():
    """Train minimal GRPO and compare GSM8K test accuracy before and after."""
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINT_DIRECTORY}")

    set_random_seed(SEED)
    policy, reference, tokenizer = load_policy_and_reference()

    # Selects only the trainable parameters (important for LoRA)
    trainable_parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    print("Loading GSM8K dataset ...")
    train_samples, val_dataset = GSM8KGRPODataset.build_train_val_datasets(
        tokenizer,
        max_prompt_len=MAX_PROMPT_LENGTH,
        val_split=VALIDATION_SPLIT,
        seed=SEED,
    )

    print("Loading GSM8K test split ...")
    test_dataset = GSM8KGRPODataset(
        tokenizer,
        max_prompt_len=MAX_PROMPT_LENGTH,
        split="test",
    )
    print(
        f"  {len(test_dataset)} samples loaded, "
        f"{test_dataset.skipped} skipped."
    )

    initial_test_accuracy = validate(
        policy,
        test_dataset,
        tokenizer,
        MAX_NEW_TOKENS,
        EVALUATION_BATCH_SIZE,
        description="test before training",
    )
    print(f"GSM8K test accuracy before training: {initial_test_accuracy:.4f}")

    if EVALUATE_EVERY != -1:
        val_accuracy = validate(
            policy,
            val_dataset,
            tokenizer,
            MAX_NEW_TOKENS,
            EVALUATION_BATCH_SIZE,
        )
        print(f"  [val] step {0:5d} | accuracy {val_accuracy:.4f}")

    terminal_columns = os.get_terminal_size(2).columns if os.isatty(2) else None
    total_steps = min(NUM_ITERATIONS, len(train_samples))
    progress = tqdm(
        total=total_steps,
        desc="train loss=----",
        unit="step",
        ncols=terminal_columns,
    )
    completed_steps = 0

    for step, sample in enumerate(train_samples):
        if step >= NUM_ITERATIONS:
            break

        prompt_tensor = torch.tensor(sample['prompt_ids'], dtype=torch.long, device="cuda").unsqueeze(0)  # [1, P]

        rollouts, rollout_masks, _ = generate_rollouts(
            policy,
            prompt_tensor,
            GROUP_SIZE,
            MAX_NEW_TOKENS,
            tokenizer,
        )  # [G, L], [G, L], [G]

        with torch.no_grad():
            old_logprobs = token_logprobs(
                policy,
                prompt_tensor,
                rollouts,
                rollout_masks
            ) # [G, L]

            reference_logprobs = token_logprobs(
                reference,
                prompt_tensor,
                rollouts,
                rollout_masks
            ) # [G, L]

        rollouts_text = tokenizer.batch_decode(rollouts, skip_special_tokens=True) # G

        ground_truth = sample["ground_truth"]
        rewards = torch.tensor(
            answer_rewards(rollouts_text, ground_truth),
            dtype=torch.float32,
            device=rollouts.device,
        ) # [G]

        advantage = (rewards - rewards.mean()) / (rewards.std(correction=0) + EPSILON) # [G]
        advantage = advantage.unsqueeze(-1) # [G, 1]

        ppo_epoch_losses = []
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()

            new_logprobs = token_logprobs(
                policy,
                prompt_tensor,
                rollouts,
                rollout_masks
            ) # [G, L]

            # Initially, new_logprobs ≈ old_logprobs, so ratio ≈ 1
            ratio = torch.exp(new_logprobs - old_logprobs) # [G, L]

            unclipped = ratio * advantage # [G, L]
            clipped = torch.clamp(ratio,
                                    1 - CLIP_EPSILON,
                                    1 + CLIP_EPSILON) * advantage # [G, L]
            surrogate = torch.minimum(unclipped, clipped) # [G, L]

            logprob_difference = reference_logprobs - new_logprobs # [G, L]

            kl_loss = torch.exp(logprob_difference) - logprob_difference - 1 # [G, L]

            grpo_loss_inside = (-surrogate + KL_COEFFICIENT * kl_loss) * rollout_masks # [G, L]

            # rollout_masks.sum() contains total number of tokens (excluding PAD )
            # clamp_min(1) just in case denominator becomes 0 with full mask sequence (edge case)
            grpo_loss = grpo_loss_inside.sum() / rollout_masks.sum().clamp_min(1)
            grpo_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            ppo_epoch_losses.append(grpo_loss.detach())

        completed_step = step + 1
        completed_steps = completed_step
        mean_loss = torch.stack(ppo_epoch_losses).mean().float().item()

        progress.set_description(f"train loss={mean_loss:.4f}")
        progress.update(1)

        if EVALUATE_EVERY > 0 and completed_step % EVALUATE_EVERY == 0:
            val_accuracy = validate(
                policy,
                val_dataset,
                tokenizer,
                MAX_NEW_TOKENS,
                EVALUATION_BATCH_SIZE,
            )
            progress.write(
                f"  [val] step {completed_step:5d} | "
                f"accuracy {val_accuracy:.4f}"
            )

        if SAVE_EVERY > 0 and completed_step % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIRECTORY,
                f"step_{completed_step:06d}",
            )
            policy.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"  [ckpt] step {completed_step:5d} -> {checkpoint_path}")

    progress.close()

    if (
        SAVE_EVERY > 0
        and completed_steps > 0
        and completed_steps % SAVE_EVERY != 0
    ):
        checkpoint_path = os.path.join(
            CHECKPOINT_DIRECTORY,
            f"step_{completed_steps:06d}",
        )
        policy.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"  [ckpt] step {completed_steps:5d} -> {checkpoint_path}")

    final_test_accuracy = validate(
        policy,
        test_dataset,
        tokenizer,
        MAX_NEW_TOKENS,
        EVALUATION_BATCH_SIZE,
        description="test after training",
    )
    print(f"GSM8K test accuracy before training: {initial_test_accuracy:.4f}")
    print(f"GSM8K test accuracy after training:  {final_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
