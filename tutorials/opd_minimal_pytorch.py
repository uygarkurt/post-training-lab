"""Minimal CUDA OPD assembled from the canonical training and xLAM code."""

import json
import os
import random
import re
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
NUM_ROLLOUTS = 2
MAX_NEW_TOKENS = 256
LEARNING_RATE = 1e-6
NUM_ITERATIONS = 2000

LORA_RANK = 8
LORA_ALPHA = 16.0

SEED = 42
VALIDATION_SPLIT = 0.005
MAX_PROMPT_LENGTH = 640
EVALUATE_EVERY = 250
EVALUATION_BATCH_SIZE = 8

SAVE_EVERY = 250
CHECKPOINT_DIRECTORY = "./checkpoints/tutorials/opd_minimal_pytorch"

DATASET_NAME = "uygarkurt/xlam-tool-calling-mixed"

ROLLOUT_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def set_random_seed(seed):
    """Seed all RNGs and require deterministic CUDA algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_policy_and_teacher():
    """Load the trainable student and a compatible frozen teacher on CUDA."""
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

    # Keep dropout disabled; evaluation mode still allows student gradients.
    policy.eval()

    print(f"Loading teacher {TEACHER_MODEL} ...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher must use the same token-to-ID mapping.")

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype="bfloat16",
    ).to("cuda")
    teacher.requires_grad_(False)
    teacher.eval()
    if (
        policy.get_output_embeddings().weight.shape[0]
        != teacher.get_output_embeddings().weight.shape[0]
    ):
        raise ValueError("Student and teacher must have the same output vocabulary size.")

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

    return policy, teacher, tokenizer


class InvalidPreparedRowError(ValueError):
    """Identify malformed rows in a prepared xLAM JSONL file."""


class ToolTemplateError(ValueError):
    """Identify tokenizers whose chat template does not expose tools."""


def parse_prepared_row(row, allow_empty_answers=False, allow_empty_tools=False):
    """Decode one prepared row, optionally allowing no calls or no tools."""
    query = row["query"]
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = (
        json.loads(row["answers"])
        if isinstance(row["answers"], str)
        else row["answers"]
    )
    if not isinstance(query, str) or not query.strip():
        raise InvalidPreparedRowError("query must be a nonempty string")
    if not isinstance(tools, list) or (not tools and not allow_empty_tools):
        raise InvalidPreparedRowError("tools must be a nonempty list")
    if not isinstance(answers, list) or (not answers and not allow_empty_answers):
        raise InvalidPreparedRowError("answers must be a nonempty list")
    if not tools and answers:
        raise InvalidPreparedRowError("a row without tools cannot contain a call")
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("function"), dict)
            or not isinstance(tool["function"].get("name"), str)
        ):
            raise InvalidPreparedRowError(
                "tools must use the standard function schema"
            )
    for answer in answers:
        if (
            not isinstance(answer, dict)
            or not isinstance(answer.get("name"), str)
            or not isinstance(answer.get("arguments"), dict)
        ):
            raise InvalidPreparedRowError(
                "answers must contain names and argument objects"
            )
    return query, tools, answers


def render_prompt(tokenizer, query, tools):
    """Render one tool-aware user prompt with the tokenizer's chat template."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if tools and tools[0]["function"]["name"] not in prompt_text:
        raise ToolTemplateError(
            "The tokenizer chat template did not render the provided tools"
        )
    return prompt_text


class XLAMFunctionCallingGRPODataset(Dataset):
    """Tokenize and hold xLAM prompts and reference calls for GRPO."""

    def __init__(
        self, tokenizer, max_prompt_len, dataset_path=None,
    ):
        """Load training rows and retain complete prompts within the limit."""
        self.samples = []
        self.total_rows = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0
        self.no_call_rows = 0
        self.retained_no_call = 0
        self.skipped_no_call_overlong = 0

        for row in hf_load_dataset(
            "json" if dataset_path else DATASET_NAME,
            data_files=str(dataset_path) if dataset_path else None,
            split="train",
        ):
            self.total_rows += 1
            try:
                query, tools, answers = parse_prepared_row(
                    row, allow_empty_answers=True,
                )
                if not answers:
                    self.no_call_rows += 1
                prompt_text = render_prompt(tokenizer, query, tools)
                prompt_ids = tokenizer.encode(prompt_text)
            except (InvalidPreparedRowError, KeyError, json.JSONDecodeError):
                self.skipped_invalid += 1
                continue
            if len(prompt_ids) > max_prompt_len:
                self.skipped_overlong += 1
                if not answers:
                    self.skipped_no_call_overlong += 1
                continue
            self.samples.append({
                "prompt_ids": prompt_ids,
                "id": row["id"],
                "query": query,
                "tools": tools,
                "answers": answers,
            })
            if not answers:
                self.retained_no_call += 1

    def __len__(self):
        """Return the number of retained training samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized prompt and its reference tool calls."""
        return self.samples[index]

    def print_summary(self, destination):
        """Print filtering statistics followed by a split description."""
        print(
            f"  {self.total_rows} rows loaded  "
            f"→  {len(self)} retained / {self.skipped_overlong} overlong / "
            f"{self.skipped_invalid} invalid  →  {destination}."
        )
        if self.no_call_rows:
            print(
                f"  No-call rows: {self.retained_no_call}/{self.no_call_rows} retained; "
                f"{self.skipped_no_call_overlong} overlong."
            )

    @classmethod
    def build_train_val_datasets(
        cls,
        tokenizer,
        max_prompt_len,
        val_split,
        seed,
        dataset_path=None,
    ):
        """Build deterministic xLAM GRPO training and validation datasets."""
        dataset = cls(
            tokenizer,
            max_prompt_len=max_prompt_len,
            dataset_path=dataset_path,
        )
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        dataset.print_summary(
            f"{len(train_dataset)} train / {len(val_dataset)} val "
            f"({val_split * 100:.0f}% val split)"
        )
        return train_dataset, val_dataset


def _normalize_predicted_call(call):
    """Normalize one common tool-call object or reject it as malformed."""
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("function"), dict):
        call = call["function"]
    name = call.get("name")
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def parse_tool_calls(completion_text):
    """Extract tool calls from tagged Qwen output or a bare JSON response."""
    tagged_calls = TOOL_CALL_PATTERN.findall(completion_text)
    if tagged_calls:
        raw_calls = []
        for payload in tagged_calls:
            try:
                raw_calls.append(json.loads(payload))
            except json.JSONDecodeError:
                return []
    else:
        payload = completion_text.strip()
        code_fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.DOTALL | re.IGNORECASE,
        )
        if code_fence:
            payload = code_fence.group(1)
        try:
            raw_calls = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list):
            return []

    normalized_calls = [_normalize_predicted_call(call) for call in raw_calls]
    if any(call is None for call in normalized_calls):
        return []
    return normalized_calls


def _canonical_call(call):
    """Serialize a call canonically, preserving JSON types."""
    return json.dumps(
        call,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tool_call_reward(completion_text, reference_calls):
    """Return parsed calls, name correctness, and the binary exact reward."""
    predicted_calls = parse_tool_calls(completion_text)

    if not reference_calls:
        payload = completion_text.strip()
        code_fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", payload,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if code_fence:
            payload = code_fence.group(1)

        if predicted_calls:
            return predicted_calls, False, False
        if "<tool_call" in completion_text:
            return predicted_calls, False, False
        if payload == "[]":
            return predicted_calls, True, True
        if payload.startswith(("{", "[")):
            return predicted_calls, False, False
        return predicted_calls, True, True

    if not predicted_calls:
        return predicted_calls, False, False

    predicted_names = Counter(call["name"] for call in predicted_calls)
    reference_names = Counter(call["name"] for call in reference_calls)
    names_correct = predicted_names == reference_names
    calls_correct = Counter(
        _canonical_call(call) for call in predicted_calls
    ) == Counter(_canonical_call(call) for call in reference_calls)
    return predicted_calls, names_correct, calls_correct


def _make_eval_collate_fn(tokenizer):
    """Build a collator that left-pads prompts and preserves call references."""
    def collate_fn(batch):
        """Pad one list of tool-aware evaluation prompts."""
        padded_prompts = tokenizer.pad(
            {"input_ids": [sample["prompt_ids"] for sample in batch]},
            padding=True,
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "prompt_ids": padded_prompts["input_ids"],
            "prompt_attention_mask": padded_prompts["attention_mask"],
            "id": [sample["id"] for sample in batch],
            "query": [sample["query"] for sample in batch],
            "tools": [sample["tools"] for sample in batch],
            "answers": [sample["answers"] for sample in batch],
        }

    return collate_fn


def build_eval_dataloader(dataset, tokenizer, batch_size):
    """Build a deterministic DataLoader of padded tool-aware prompts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_eval_collate_fn(tokenizer),
        num_workers=0,
        drop_last=False,
    )


class XLAMFunctionCallingEvalDataset(Dataset):
    """Tokenize xLAM test prompts with their available tools."""

    def __init__(
        self,
        tokenizer,
        max_prompt_len,
        num_samples=None,
        dataset_path=None,
    ):
        """Load test rows, optionally retaining only the first N usable rows."""
        self.samples = []
        self.skipped = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0

        for row in hf_load_dataset(
            "json" if dataset_path else DATASET_NAME,
            data_files={"test": str(dataset_path)} if dataset_path else None,
            split="test",
        ):
            try:
                query, tools, answers = parse_prepared_row(
                    row, allow_empty_answers=True, allow_empty_tools=True,
                )
                prompt_text = render_prompt(tokenizer, query, tools)
                prompt_ids = tokenizer.encode(prompt_text)
            except (InvalidPreparedRowError, KeyError, json.JSONDecodeError):
                self.skipped_invalid += 1
                self.skipped += 1
                continue
            if len(prompt_ids) > max_prompt_len:
                self.skipped_overlong += 1
                self.skipped += 1
                continue
            self.samples.append({
                "prompt_ids": prompt_ids,
                "id": row["id"],
                "query": query,
                "tools": tools,
                "answers": answers,
            })

        if num_samples is not None:
            self.samples = self.samples[:num_samples]

    def __len__(self):
        """Return the number of retained test samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized prompt and its reference tool calls."""
        return self.samples[index]


def validate(
    policy,
    val_dataset,
    tokenizer,
    max_new_tokens,
    batch_size,
    description="val",
):
    """Measure exact tool-call and function-name accuracy with greedy decoding."""
    correct_calls = 0
    correct_names = 0
    total_answers = 0
    device = next(policy.parameters()).device
    val_loader = build_eval_dataloader(
        val_dataset,
        tokenizer=tokenizer,
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
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_length = prompt_tensor.shape[1]
            completion_ids = generated_ids[:, prompt_length:]
            completion_texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
            for index, text in enumerate(completion_texts):
                _, names_correct, calls_correct = (
                    tool_call_reward(
                        text,
                        batch["answers"][index],
                    )
                )
                correct_names += names_correct
                correct_calls += calls_correct
            total_answers += len(batch["answers"])

    if total_answers == 0:
        return {"accuracy": float("nan"), "name_accuracy": float("nan")}
    print(
        f"  {correct_calls}/{total_answers} exact call sets; "
        f"{correct_names}/{total_answers} function-name sets."
    )
    return {
        "accuracy": correct_calls / total_answers,
        "name_accuracy": correct_names / total_answers,
    }


def validate_policy(policy, val_dataset, tokenizer, step, progress=None):
    """Run and log the validation metrics for the selected prompt dataset."""
    metrics = validate(
        policy,
        val_dataset,
        tokenizer,
        MAX_NEW_TOKENS,
        EVALUATION_BATCH_SIZE,
    )
    message = (
        f"  [val] step {step:5d} | exact call accuracy "
        f"{metrics['accuracy']:.4f} | name accuracy "
        f"{metrics['name_accuracy']:.4f}"
    )
    if progress is None:
        print(message)
    else:
        progress.write(message)


@torch.no_grad()
def generate_rollouts(policy, prompt_tensor, num_rollouts, max_new_tok, tokenizer):
    """Sample student completions and mask tokens after their first EOS or PAD."""
    # All allowing mask. Removes EOS and PAD token ambiguity.
    prompt_attention_mask = torch.ones_like(prompt_tensor)
    generated_ids = policy.generate(
        input_ids=prompt_tensor,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tok,
        num_return_sequences=num_rollouts,
        **ROLLOUT_SAMPLING,
        use_cache=True
    )  # [G, P + L]

    prompt_length = prompt_tensor.shape[1]  # P
    completion_ids = generated_ids[:, prompt_length:]  # [G, L]

    # Works for Qwen. It produces EOS after generation ends. Fills the rest with PAD.
    is_eos_or_pad = (completion_ids == tokenizer.eos_token_id) | (completion_ids == tokenizer.pad_token_id)

    # We have to include the first EOS token
    completion_mask = (is_eos_or_pad.cumsum(dim=-1) <= 1).long()  # [G, L]

    return completion_ids, completion_mask


def save_checkpoint(
    policy, tokenizer, checkpoint_dir, step, training_start_time, learning_rate,
):
    """Save weights, tokenizer, elapsed hours, and the learning rate."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:06d}")
    policy.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    metadata = {
        "step": step,
        "elapsed_hours": (time.monotonic() - training_start_time) / 3600,
        "learning_rate": learning_rate,
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"  [ckpt] step {step:5d} -> {checkpoint_path}")


def main():
    """Train OPD and compare xLAM test accuracy before and after.

    Tensor dimension notation:
        N: Number of student rollouts per prompt.
        P: Number of tokens in the prompt.
        L: Number of generated completion tokens, excluding the prompt.
        V: Number of tokens in the model vocabulary.
    """
    saved_args = {
        "MODEL": MODEL,
        "TEACHER_MODEL": TEACHER_MODEL,
        "NUM_ROLLOUTS": NUM_ROLLOUTS,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_ITERATIONS": NUM_ITERATIONS,
        "LORA_RANK": LORA_RANK,
        "LORA_ALPHA": LORA_ALPHA,
        "SEED": SEED,
        "VALIDATION_SPLIT": VALIDATION_SPLIT,
        "MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH,
        "EVALUATE_EVERY": EVALUATE_EVERY,
        "EVALUATION_BATCH_SIZE": EVALUATION_BATCH_SIZE,
        "SAVE_EVERY": SAVE_EVERY,
        "CHECKPOINT_DIRECTORY": CHECKPOINT_DIRECTORY,
        "DATASET_NAME": DATASET_NAME,
        "ROLLOUT_SAMPLING": ROLLOUT_SAMPLING,
    }
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"opd_{timestamp}"
    checkpoint_dir = os.path.join(
        os.path.normpath(CHECKPOINT_DIRECTORY),
        run_name,
    )
    run_logs_dir = os.path.join(checkpoint_dir, "run_logs")
    os.makedirs(run_logs_dir)

    with open(os.path.join(run_logs_dir, "args.json"), "w") as args_file:
        json.dump(saved_args, args_file, indent=2)

    terminal_columns = os.get_terminal_size(2).columns if os.isatty(2) else None
    print(f"Run logs directory: {run_logs_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Rollout sampling: {ROLLOUT_SAMPLING}")

    set_random_seed(SEED)
    policy, teacher, tokenizer = load_policy_and_teacher()
    trainable_parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE)
    print("Loading xlam-tool-calling-mixed dataset ...")
    train_samples, val_dataset = XLAMFunctionCallingGRPODataset.build_train_val_datasets(
        tokenizer,
        max_prompt_len=MAX_PROMPT_LENGTH,
        val_split=VALIDATION_SPLIT,
        seed=SEED,
    )

    print("Loading xLAM/irrelevance function-calling test split ...")
    test_dataset = XLAMFunctionCallingEvalDataset(
        tokenizer,
        max_prompt_len=MAX_PROMPT_LENGTH,
    )
    print(
        f"  {len(test_dataset)} samples loaded, "
        f"{test_dataset.skipped} skipped."
    )

    initial_test_metrics = validate(
        policy,
        test_dataset,
        tokenizer,
        MAX_NEW_TOKENS,
        EVALUATION_BATCH_SIZE,
        description="test before training",
    )
    print(
        "xLAM test exact tool-call accuracy before training: "
        f"{initial_test_metrics['accuracy']:.4f}"
    )
    print(
        "xLAM test function-name accuracy before training: "
        f"{initial_test_metrics['name_accuracy']:.4f}"
    )

    if EVALUATE_EVERY != -1:
        validate_policy(policy, val_dataset, tokenizer, step=0)

    batches_per_epoch = len(train_samples)
    total_steps = NUM_ITERATIONS if batches_per_epoch else 0
    learning_rate_display = f"{optimizer.param_groups[0]['lr']:.3g}".replace("e-0", "e-")
    progress = tqdm(
        total=total_steps,
        desc=(
            f"epoch=1 batch=0/{batches_per_epoch} loss=---- "
            f"lr={learning_rate_display}"
        ),
        unit="step",
        ncols=terminal_columns,
    )
    completed_steps = 0
    training_start_time = time.monotonic()

    for step in range(total_steps):
        epoch_number, batch_offset = divmod(step, batches_per_epoch)
        batch_index = batch_offset + 1
        sample = train_samples[batch_offset]
        prompt_tensor = torch.tensor(
            sample["prompt_ids"], dtype=torch.long, device="cuda",
        ).unsqueeze(0)  # [1, P]

        rollouts, rollout_masks = generate_rollouts(
            policy,
            prompt_tensor,
            NUM_ROLLOUTS,
            MAX_NEW_TOKENS,
            tokenizer,
        )  # [N, L], [N, L]

        prompt_ids_extended = torch.repeat_interleave(prompt_tensor, repeats=rollouts.size(0), dim=0)  # [N, P]
        complete_ids = torch.cat([prompt_ids_extended, rollouts], dim=-1)  # [N, P + L]

        student_logits = policy(complete_ids, use_cache=False).logits  # [N, P + L, V]

        # Take the last prompt token, the rest is regular logits shift
        student_logits_shift = student_logits[:, prompt_tensor.size(1) - 1:-1, :]  # [N, L, V]

        with torch.no_grad():
            teacher_logits = teacher(complete_ids, use_cache=False).logits  # [N, P + L, V]
            teacher_logits_shift = teacher_logits[:, prompt_tensor.size(1) - 1:-1, :]  # [N, L, V]

        student_logprobs = torch.log_softmax(
            student_logits_shift, dim=-1, dtype=torch.float32,
        )  # [N, L, V]
        teacher_logprobs = torch.log_softmax(
            teacher_logits_shift, dim=-1, dtype=torch.float32,
        )  # [N, L, V]

        # Summing over V gives KL div values for every position
        kl = F.kl_div(
            teacher_logprobs,
            student_logprobs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)  # [N, L]

        opd_loss = (kl * rollout_masks).sum() / rollout_masks.sum()

        optimizer.zero_grad()
        opd_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        completed_step = step + 1
        completed_steps = completed_step
        mean_loss = opd_loss.detach().float().item()
        learning_rate_display = f"{optimizer.param_groups[0]['lr']:.3g}".replace("e-0", "e-")
        progress.set_description(
            f"epoch={epoch_number + 1} batch={batch_index}/{batches_per_epoch} "
            f"loss={mean_loss:.4f} lr={learning_rate_display}",
            refresh=False,
        )
        progress.update(1)

        if EVALUATE_EVERY > 0 and completed_step % EVALUATE_EVERY == 0:
            validate_policy(
                policy, val_dataset, tokenizer, completed_step,
                progress=progress,
            )

        if SAVE_EVERY > 0 and completed_step % SAVE_EVERY == 0:
            save_checkpoint(
                policy, tokenizer, checkpoint_dir, completed_step,
                training_start_time, optimizer.param_groups[0]["lr"],
            )

    progress.close()

    if (
        SAVE_EVERY > 0
        and completed_steps > 0
        and completed_steps % SAVE_EVERY != 0
    ):
        save_checkpoint(
            policy, tokenizer, checkpoint_dir, completed_steps,
            training_start_time, optimizer.param_groups[0]["lr"],
        )

    final_test_metrics = validate(
        policy,
        test_dataset,
        tokenizer,
        MAX_NEW_TOKENS,
        EVALUATION_BATCH_SIZE,
        description="test after training",
    )
    print(
        "xLAM test exact tool-call accuracy after training:  "
        f"{final_test_metrics['accuracy']:.4f}"
    )
    print(
        "xLAM test function-name accuracy after training:  "
        f"{final_test_metrics['name_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
