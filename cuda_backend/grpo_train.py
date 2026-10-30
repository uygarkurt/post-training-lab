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

from cuda_backend import gsm8k_eval, xlam_function_calling_eval
from data_preparation import gsm8k, xlam_function_calling

ROLLOUT_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training on CUDA (NVIDIA GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Overfit a tiny subset (same samples for train and val; real task reward)",
    )
    parser.add_argument(
        "--debug-samples", type=int, default=8,
        help="Number of samples in --debug mode (train and val use the same set)",
    )
    parser.add_argument(
        "--dataset",
        choices=("xlam-tool-calling-mixed", "gsm8k"),
        default="gsm8k",
        help="GRPO dataset and matching verifiable reward",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
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
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from the selected training dataset")
    parser.add_argument("--max-prompt-len", type=int, default=640, help="Skip selected-dataset prompts longer than this")
    parser.add_argument("--log-every", type=int, default=10, help="Write aggregated training metrics to TensorBoard every N steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Prompts generated together during validation")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/cuda/grpo", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    if args.ppo_epochs < 1:
        parser.error("--ppo-epochs must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
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
    """Load the selected GRPO dataset with a deterministic runtime split."""
    if args.dataset == "xlam-tool-calling-mixed":
        print("Loading xlam-tool-calling-mixed dataset ...")
        dataset_class = xlam_function_calling.XLAMFunctionCallingGRPODataset
    else:
        print("Loading GSM8K dataset ...")
        dataset_class = gsm8k.GSM8KGRPODataset

    if args.debug:
        return dataset_class.build_debug_overfit_datasets(
            tokenizer,
            max_prompt_len=args.max_prompt_len,
            seed=args.seed,
            debug_samples=args.debug_samples,
        )

    return dataset_class.build_train_val_datasets(
        tokenizer,
        max_prompt_len=args.max_prompt_len,
        val_split=args.val_split,
        seed=args.seed,
    )


@torch.no_grad()
def generate_rollouts(policy, prompt_tensor, group_size, max_new_tok, tokenizer):
    # All allowing mask. Removes EOS and PAD token ambiguity.
    prompt_attention_mask = torch.ones_like(prompt_tensor)

    generated_ids = policy.generate(
        input_ids=prompt_tensor,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tok,
        num_return_sequences=group_size,
        # Match the unmodified softmax distribution scored by token_logprobs.
        # Override Qwen's inherited top-k and repetition penalty explicitly.
        **ROLLOUT_SAMPLING,
        use_cache=True,
    )  # [G, P + L]

    prompt_length = prompt_tensor.shape[1]  # P
    completion_ids = generated_ids[:, prompt_length:]  # [G, L]

    # Works for Qwen. It produces EOS after generation ends. Fills the rest with PAD.
    is_eos_or_pad = (completion_ids == tokenizer.eos_token_id) | (completion_ids == tokenizer.pad_token_id)

    # We have to include the first EOS token
    completion_mask = (is_eos_or_pad.cumsum(dim=-1) <= 1).long()  # [G, L]
    truncated_rollouts = ~is_eos_or_pad.any(dim=-1)  # [G]

    return completion_ids, completion_mask, truncated_rollouts


def token_logprobs(model, prompt_ids, rollouts, rollout_masks):
    prompt_ids_extended = prompt_ids.repeat_interleave(rollouts.shape[0], dim=0)  # [G, P]
    complete_ids = torch.cat([prompt_ids_extended, rollouts], dim=-1)  # [G, P + L]

    # Avoid materializing unused prompt logits on models that support this.
    # The final prompt token predicts the first completion token.
    model_kwargs = {}
    if model._supports_logits_to_keep():
        model_kwargs["logits_to_keep"] = rollouts.shape[1] + 1
    logits = model(complete_ids, use_cache=False, **model_kwargs).logits

    # Take the completion predictors, excluding the last (out-of-scope) token.
    # Negative indexing also handles models that return the full sequence.
    logits_shift = logits[:, -(rollouts.shape[1] + 1):-1, :]  # [G, L, V]

    # Keep ratios and KL in float32 even when model logits are bfloat16.
    logprobs = torch.log_softmax(logits_shift, dim=-1, dtype=torch.float32)  # [G, L, V]

    # Choose probabilities that corresponds to tokens in the rollout with fancy indexing.
    group_indices = torch.arange(rollouts.shape[0], device=rollouts.device).unsqueeze(-1)
    position_indices = torch.arange(rollouts.shape[1], device=rollouts.device).unsqueeze(0)
    selected_logprobs = logprobs[group_indices, position_indices, rollouts]  # [G, L]

    # Apply mask
    selected_logprobs_masked = (
        selected_logprobs * rollout_masks
    )

    return selected_logprobs_masked


def group_advantages(rewards, epsilon):
    """Standardize group rewards, giving constant groups exactly zero advantage."""
    reward_std = rewards.std(correction=0)
    mixed_group = rewards.max() > rewards.min()
    # Repeated fractional rewards can have a rounded mean unequal to each
    # element. Such constant groups must have exactly zero task advantage.
    return torch.where(
        mixed_group,
        (rewards - rewards.mean()) / (reward_std + epsilon),
        torch.zeros_like(rewards),
    ) # [G]


def validate_policy(policy, val_dataset, tokenizer, args, writer, step, progress=None):
    """Run and log the validation metrics for the selected GRPO dataset."""
    if args.dataset == "xlam-tool-calling-mixed":
        metrics = xlam_function_calling_eval.validate(
            policy,
            val_dataset,
            tokenizer,
            args.max_new_tok,
            args.eval_batch_size,
        )
        writer.add_scalar("val/accuracy", metrics["accuracy"], step)
        writer.add_scalar("val/name_accuracy", metrics["name_accuracy"], step)
        call_accuracy = metrics.get("call_accuracy", float("nan"))
        no_call_accuracy = metrics.get("no_call_accuracy", float("nan"))
        irrelevant_tool_accuracy = metrics.get(
            "irrelevant_tool_accuracy", float("nan"),
        )
        if not np.isnan(call_accuracy):
            writer.add_scalar("val/call_accuracy", call_accuracy, step)
        if not np.isnan(no_call_accuracy):
            writer.add_scalar(
                "val/no_call_accuracy", no_call_accuracy, step,
            )
        if not np.isnan(irrelevant_tool_accuracy):
            writer.add_scalar(
                "val/irrelevant_tool_accuracy",
                irrelevant_tool_accuracy,
                step,
            )
        message = (
            f"  [val] step {step:5d} | exact call accuracy "
            f"{metrics['accuracy']:.4f} | name accuracy "
            f"{metrics['name_accuracy']:.4f}"
        )
        if not np.isnan(no_call_accuracy):
            message += (
                f" | call accuracy {call_accuracy:.4f}"
                f" | no-call accuracy {no_call_accuracy:.4f}"
            )
    else:
        accuracy = gsm8k_eval.validate(
            policy,
            val_dataset,
            tokenizer,
            args.max_new_tok,
            args.eval_batch_size,
        )
        writer.add_scalar("val/accuracy", accuracy, step)
        message = f"  [val] step {step:5d} | accuracy {accuracy:.4f}"

    if progress is None:
        print(message)
    else:
        progress.write(message)


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
    """
    Tensor dimension notation:
        G: Number of generated rollouts in a group.
        P: Number of tokens in the prompt.
        L: Number of generated completion tokens, excluding the prompt.
        V: Number of tokens in the model vocabulary.
    """
    args = parse_args()
    saved_args = vars(args).copy()
    saved_args["rollout_sampling"] = ROLLOUT_SAMPLING.copy()
    saved_args["reward_function"] = (
        "xlam_binary"
        if args.dataset == "xlam-tool-calling-mixed" else "gsm8k_exact"
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"grpo_{timestamp}"
    args.checkpoint_dir = os.path.join(
        os.path.normpath(args.checkpoint_dir),
        run_name,
    )
    run_logs_dir = os.path.join(args.checkpoint_dir, "run_logs")
    os.makedirs(run_logs_dir)

    with open(os.path.join(run_logs_dir, "args.json"), "w") as args_file:
        json.dump(saved_args, args_file, indent=2)

    terminal_columns = os.get_terminal_size(2).columns if os.isatty(2) else None
    terminal_stdout = os.dup(1)
    terminal_stderr = os.dup(2)
    tee_process = subprocess.Popen(
        ["tee", os.path.join(run_logs_dir, "out.log")],
        stdin=subprocess.PIPE,
    )
    os.dup2(tee_process.stdin.fileno(), 1)
    os.dup2(tee_process.stdin.fileno(), 2)

    print(f"Run logs directory: {run_logs_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Training reward: {saved_args['reward_function']}")
    print(f"Rollout sampling: {ROLLOUT_SAMPLING}")

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
    writer = SummaryWriter(log_dir=run_logs_dir)
    print(f"TensorBoard logs: tensorboard --logdir={run_logs_dir}")

    if args.eval_every != -1:
        validate_policy(
            policy,
            val_dataset,
            tokenizer,
            args,
            writer,
            step=0,
        )

    batches_per_epoch = len(train_samples)
    total_steps = args.num_iters if batches_per_epoch else 0
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
    log_step_count = 0
    loss_sum_since_log = 0.0
    reward_sum_since_log = 0.0
    exact_reward_sum_since_log = 0.0
    truncated_rollout_fraction_sum_since_log = 0.0
    all_zero_group_count_since_log = 0
    all_one_group_count_since_log = 0
    mixed_group_count_since_log = 0
    kl_sum_since_log = 0.0
    gradient_norm_sum_since_log = 0.0
    completion_tokens_sum_since_log = 0
    elapsed_time_since_log = 0.0

    for step in range(total_steps):
        epoch_number, batch_offset = divmod(step, batches_per_epoch)
        batch_index = batch_offset + 1
        sample = train_samples[batch_offset]
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

        if args.dataset == "xlam-tool-calling-mixed":
            reward_values = []
            for rollout_text in rollouts_text:
                _, _, reward_is_one = xlam_function_calling.tool_call_reward(
                    rollout_text,
                    sample["answers"],
                )
                reward_values.append(float(reward_is_one))
            rewards = torch.tensor(
                reward_values,
                dtype=torch.float32,
                device=rollouts.device,
            ) # [G]
        else:
            rewards = torch.tensor(
                gsm8k.answer_rewards(
                    rollouts_text,
                    sample["ground_truth"],
                ),
                dtype=torch.float32,
                device=rollouts.device,
            ) # [G]

        mixed_group = rewards.max() > rewards.min()
        advantage = group_advantages(rewards, args.epsilon) # [G]
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

            # expm1 avoids subtracting nearly equal values when the policies agree.
            kl_loss = torch.expm1(logprob_difference) - logprob_difference # [G, L]

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
        # Both reward variants reserve 1.0 for exact task correctness.
        mean_exact_reward = rewards.eq(1.0).float().mean().item()
        truncated_rollout_fraction = truncated_rollouts.float().mean().item()
        completion_tokens = rollout_masks.sum().item()
        step_elapsed_time = time.time() - step_start_time

        log_step_count += 1
        loss_sum_since_log += mean_loss
        reward_sum_since_log += mean_reward
        exact_reward_sum_since_log += mean_exact_reward
        truncated_rollout_fraction_sum_since_log += truncated_rollout_fraction
        all_zero_group_count_since_log += int(rewards.eq(0.0).all().item())
        all_one_group_count_since_log += int(rewards.eq(1.0).all().item())
        mixed_group_count_since_log += int(mixed_group.item())
        kl_sum_since_log += mean_kl
        gradient_norm_sum_since_log += mean_gradient_norm
        completion_tokens_sum_since_log += completion_tokens
        elapsed_time_since_log += step_elapsed_time

        if completed_step % args.log_every == 0 or completed_step == total_steps:
            writer.add_scalar(
                "train/loss",
                loss_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/reward",
                reward_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/exact_reward",
                exact_reward_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/truncated_rollout_fraction",
                truncated_rollout_fraction_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/all_zero_group_fraction",
                all_zero_group_count_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/all_one_group_fraction",
                all_one_group_count_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/mixed_group_fraction",
                mixed_group_count_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/kl",
                kl_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/grad_norm",
                gradient_norm_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar("train/learning_rate", args.lr, completed_step)
            writer.add_scalar(
                "train/completion_tokens",
                completion_tokens_sum_since_log / log_step_count,
                completed_step,
            )
            writer.add_scalar(
                "train/tokens_per_sec",
                completion_tokens_sum_since_log / max(elapsed_time_since_log, 1e-8),
                completed_step,
            )

            log_step_count = 0
            loss_sum_since_log = 0.0
            reward_sum_since_log = 0.0
            exact_reward_sum_since_log = 0.0
            truncated_rollout_fraction_sum_since_log = 0.0
            all_zero_group_count_since_log = 0
            all_one_group_count_since_log = 0
            mixed_group_count_since_log = 0
            kl_sum_since_log = 0.0
            gradient_norm_sum_since_log = 0.0
            completion_tokens_sum_since_log = 0
            elapsed_time_since_log = 0.0

        learning_rate_display = f"{optimizer.param_groups[0]['lr']:.3g}".replace("e-0", "e-")
        progress.set_description(
            f"epoch={epoch_number + 1} batch={batch_index}/{batches_per_epoch} "
            f"loss={mean_loss:.4f} lr={learning_rate_display}",
            refresh=False,
        )
        progress.update(1)

        if args.eval_every > 0 and completed_step % args.eval_every == 0:
            validate_policy(
                policy,
                val_dataset,
                tokenizer,
                args,
                writer,
                completed_step,
                progress=progress,
            )

        if args.save_every > 0 and completed_step % args.save_every == 0:
            writer.flush()
            save_checkpoint(
                policy,
                tokenizer,
                args.checkpoint_dir,
                completed_step,
                training_start_time,
                optimizer.param_groups[0]["lr"],
            )

    progress.close()

    if (
        args.save_every > 0
        and completed_steps > 0
        and completed_steps % args.save_every != 0
    ):
        writer.flush()
        save_checkpoint(
            policy,
            tokenizer,
            args.checkpoint_dir,
            completed_steps,
            training_start_time,
            optimizer.param_groups[0]["lr"],
        )

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
