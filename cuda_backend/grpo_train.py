import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np
import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cuda_backend import gsm8k_eval
from data_preparation import gsm8k


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training on CUDA (NVIDIA GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Overfit a tiny GSM8K subset (same samples for train and val; real answer reward)",
    )
    parser.add_argument(
        "--debug-samples", type=int, default=8,
        help="Number of GSM8K samples in --debug mode (train and val use the same set)",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Hugging Face base model, merged model, or local model directory",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Optional PEFT adapter checkpoint used to initialize the model",
    )
    parser.add_argument(
        "--train-mode", choices=("lora", "full"), default="lora",
        help="Train LoRA parameters only, or all parameters of a dense model",
    )

    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt (G)")
    parser.add_argument("--max-new-tok", type=int, default=256, help="Max tokens to generate per rollout")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO inner epochs per step")
    parser.add_argument("--num-iters", type=int, default=500, help="Number of outer training iterations")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Advantage normalisation epsilon")

    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale = alpha / rank)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization, data shuffle, and rollout sampling")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from GSM8K train set")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Skip GSM8K prompts longer than this")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Prompts generated together during validation")

    parser.add_argument("--tensorboard-dir", type=str, default="./runs/cuda/grpo", help="Base path for timestamped TensorBoard run directories")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/cuda/grpo", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    if args.ppo_epochs < 1:
        parser.error("--ppo-epochs must be at least 1")
    return args


def set_random_seed(seed):
    """Seed all RNGs and require deterministic CUDA algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_policy_and_reference(args):
    """
    Load the initial GRPO policy and an identical frozen reference model.

    An existing adapter is continued in LoRA mode. In full mode, it is first
    merged into the base model so the resulting policy has a normal dense
    parameterization and can be saved as a complete model.
    """
    print(f"Loading model {args.model} ...")
    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load an existing adapter, then either continue LoRA or merge for full training.
    if args.adapter:
        adapter_config = PeftConfig.from_pretrained(args.adapter)
        adapter_base = adapter_config.base_model_name_or_path
        if adapter_base and adapter_base != args.model:
            print(
                "Warning: the adapter reports base model "
                f"'{adapter_base}', but --model is '{args.model}'."
            )

        continue_adapter_training = args.train_mode == "lora"
        print(f"Loading adapter {args.adapter} ...")
        policy = PeftModel.from_pretrained(
            policy,
            args.adapter,
            is_trainable=continue_adapter_training,
        )

        # Full training starts from the adapter's merged dense weights.
        if args.train_mode == "full":
            print("Merging adapter into the model for full training ...")
            policy = policy.merge_and_unload()
            policy.requires_grad_(True)

    # LoRA training without an adapter attaches a new adapter to the full model.
    elif args.train_mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        )
        print("Applying new LoRA adapters to all linear layers ...")
        policy = get_peft_model(policy, lora_config)

    # Full training without an adapter updates every parameter of the full model.
    else:
        policy.requires_grad_(True)

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
        f"Training mode: {args.train_mode} | "
        f"{trainable_parameters:,} / {total_parameters:,} parameters trainable"
    )

    return policy, reference, tokenizer


def load_grpo_dataset(tokenizer, args):
    print("Loading GSM8K dataset ...")
    if args.debug:
        return gsm8k.GSM8KGRPODataset.build_debug_overfit_datasets(
            tokenizer,
            max_prompt_len=args.max_prompt_len,
            seed=args.seed,
            debug_samples=args.debug_samples,
        )

    return gsm8k.GSM8KGRPODataset.build_train_val_datasets(
        tokenizer,
        max_prompt_len=args.max_prompt_len,
        val_split=args.val_split,
        seed=args.seed,
    )


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
    args = parse_args()
    saved_args = vars(args).copy()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    args.tensorboard_dir = f"{os.path.normpath(args.tensorboard_dir)}_{timestamp}"
    args.checkpoint_dir = f"{os.path.normpath(args.checkpoint_dir)}_{timestamp}"
    os.makedirs(args.tensorboard_dir)
    os.makedirs(args.checkpoint_dir)

    with open(os.path.join(args.tensorboard_dir, "args.json"), "w") as args_file:
        json.dump(saved_args, args_file, indent=2)

    terminal_columns = os.get_terminal_size(2).columns if os.isatty(2) else None
    terminal_stdout = os.dup(1)
    terminal_stderr = os.dup(2)
    tee_process = subprocess.Popen(
        ["tee", os.path.join(args.tensorboard_dir, "out.log")],
        stdin=subprocess.PIPE,
    )
    os.dup2(tee_process.stdin.fileno(), 1)
    os.dup2(tee_process.stdin.fileno(), 2)

    print(f"Run directory: {args.tensorboard_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    set_random_seed(args.seed)
    policy, reference, tokenizer = load_policy_and_reference(args)

    # Selects only the trainable parameters (important for LoRA)
    trainable_parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
    )

    train_samples, val_dataset = load_grpo_dataset(tokenizer, args)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs: tensorboard --logdir={args.tensorboard_dir}")

    if args.eval_every != -1:
        val_accuracy = gsm8k_eval.validate(
            policy,
            val_dataset,
            tokenizer,
            args.max_new_tok,
            args.eval_batch_size,
        )
        writer.add_scalar("val/accuracy", val_accuracy, 0)
        print(f"  [val] step {0:5d} | accuracy {val_accuracy:.4f}")

    total_steps = min(args.num_iters, len(train_samples))
    progress = tqdm(
        total=total_steps,
        desc="train loss=----",
        unit="step",
        ncols=terminal_columns,
    )
    completed_steps = 0

    for step, sample in enumerate(train_samples):
        if step >= args.num_iters:
            break
        step_start_time = time.time()

        prompt_tensor = torch.tensor(sample['prompt_ids'], dtype=torch.long, device="cuda").unsqueeze(0)  # [1, P]

        rollouts, rollout_masks, truncated_rollouts = generate_rollouts(
            policy,
            prompt_tensor,
            args.group_size,
            args.max_new_tok,
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
            gsm8k.answer_rewards(rollouts_text, ground_truth),
            dtype=torch.float32,
            device=rollouts.device,
        ) # [G]

        advantage = (rewards - rewards.mean()) / (rewards.std(correction=0) + args.epsilon) # [G]
        advantage = advantage.unsqueeze(-1) # [G, 1]

        ppo_epoch_losses = []
        ppo_epoch_kl_values = []
        ppo_epoch_gradient_norms = []
        for _ in range(args.ppo_epochs):
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
                                    1 - args.clip_eps,
                                    1 + args.clip_eps) * advantage # [G, L]
            surrogate = torch.minimum(unclipped, clipped) # [G, L]

            logprob_difference = reference_logprobs - new_logprobs # [G, L]

            kl_loss = torch.exp(logprob_difference) - logprob_difference - 1 # [G, L]

            grpo_loss_inside = (-surrogate + args.kl_coef * kl_loss) * rollout_masks # [G, L]

            # rollout_masks.sum() contains total number of tokens (excluding PAD )
            # clamp_min(1) just in case denominator becomes 0 with full mask sequence (edge case)
            grpo_loss = grpo_loss_inside.sum() / rollout_masks.sum().clamp_min(1)
            mean_kl = (kl_loss * rollout_masks).sum() / rollout_masks.sum().clamp_min(1)
            grpo_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            ppo_epoch_losses.append(grpo_loss.detach())
            ppo_epoch_kl_values.append(mean_kl.detach())
            ppo_epoch_gradient_norms.append(gradient_norm.detach())

        completed_step = step + 1
        completed_steps = completed_step
        mean_loss = torch.stack(ppo_epoch_losses).mean().float().item()
        mean_kl = torch.stack(ppo_epoch_kl_values).mean().float().item()
        mean_gradient_norm = torch.stack(
            ppo_epoch_gradient_norms
        ).mean().float().item()
        mean_reward = rewards.mean().item()
        reward_std = rewards.std(correction=0).item()
        truncated_rollout_fraction = truncated_rollouts.float().mean().item()
        completion_tokens = rollout_masks.sum().item()
        tokens_per_second = completion_tokens / max(
            time.time() - step_start_time,
            1e-8,
        )

        writer.add_scalar("train/loss", mean_loss, completed_step)
        writer.add_scalar("train/reward", mean_reward, completed_step)
        writer.add_scalar("train/reward_std", reward_std, completed_step)
        writer.add_scalar(
            "train/truncated_rollout_fraction",
            truncated_rollout_fraction,
            completed_step,
        )
        writer.add_scalar(
            "train/informative_group",
            float(reward_std > 0.0),
            completed_step,
        )
        writer.add_scalar("train/kl", mean_kl, completed_step)
        writer.add_scalar("train/grad_norm", mean_gradient_norm, completed_step)
        writer.add_scalar("train/learning_rate", args.lr, completed_step)
        writer.add_scalar(
            "train/completion_tokens",
            completion_tokens,
            completed_step,
        )
        writer.add_scalar(
            "train/tokens_per_sec",
            tokens_per_second,
            completed_step,
        )

        progress.set_description(f"train loss={mean_loss:.4f}")
        progress.update(1)

        if args.eval_every > 0 and completed_step % args.eval_every == 0:
            val_accuracy = gsm8k_eval.validate(
                policy,
                val_dataset,
                tokenizer,
                args.max_new_tok,
                args.eval_batch_size,
            )
            writer.add_scalar("val/accuracy", val_accuracy, completed_step)
            progress.write(
                f"  [val] step {completed_step:5d} | "
                f"accuracy {val_accuracy:.4f}"
            )

        if args.save_every > 0 and completed_step % args.save_every == 0:
            writer.flush()
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"step_{completed_step:06d}",
            )
            policy.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"  [ckpt] step {completed_step:5d} -> {checkpoint_path}")

    progress.close()

    if (
        args.save_every > 0
        and completed_steps > 0
        and completed_steps % args.save_every != 0
    ):
        writer.flush()
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"step_{completed_steps:06d}",
        )
        policy.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"  [ckpt] step {completed_steps:5d} -> {checkpoint_path}")

    writer.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(terminal_stdout, 1)
    os.dup2(terminal_stderr, 2)
    os.close(terminal_stdout)
    os.close(terminal_stderr)
    tee_process.stdin.close()
    tee_process.wait()


if __name__ == "__main__":
    main()
