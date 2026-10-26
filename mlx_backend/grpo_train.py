import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
from mlx_lm.utils import save_config, save_model
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_preparation import gsm8k, xlam_function_calling
from mlx_backend import gsm8k_eval, xlam_function_calling_eval

ROLLOUT_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}


def parse_args():
    """Parse the canonical GRPO training options."""
    parser = argparse.ArgumentParser(
        description="GRPO training on MLX (Apple Silicon).",
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
        help="Optional native MLX adapter checkpoint used to initialize the model",
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
    """Seed Python, NumPy, PyTorch data loading, and MLX."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)


def load_policy_and_reference(args):
    """
    Load the initial GRPO policy and an identical frozen reference model.

    An existing native MLX adapter is continued in LoRA mode. In full mode,
    it is first merged into the base model so the resulting policy has a normal
    dense parameterization and can be saved as a complete model.
    """
    print(f"Loading model {args.model} ...")
    policy, tokenizer, model_config = load(args.model, return_config=True)
    policy.set_dtype(mx.bfloat16)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    adapter_config = None
    # Load an existing adapter, then either continue LoRA or merge for full training.
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.is_dir():
            adapter_path = Path(snapshot_download(args.adapter))
        with open(adapter_path / "adapter_config.json") as adapter_file:
            adapter_config = json.load(adapter_file)
        adapter_base = adapter_config.get("base_model")
        if adapter_base and adapter_base != args.model:
            print(
                "Warning: the adapter reports base model "
                f"'{adapter_base}', but --model is '{args.model}'."
            )

        print(f"Loading adapter {args.adapter} ...")
        policy.freeze()
        load_adapters(policy, adapter_path)

        # Full training starts from the adapter's merged dense weights.
        if args.train_mode == "full":
            print("Merging adapter into the model for full training ...")
            policy.update_modules(tree_unflatten([
                (name, module.fuse())
                for name, module in policy.named_modules()
                if hasattr(module, "fuse")
            ]))
            policy.unfreeze()
            adapter_config = None

    # LoRA training without an adapter attaches a new adapter to the full model.
    elif args.train_mode == "lora":
        # Explicit keys match PEFT's all-linear targets, excluding the output head.
        lora_config = {
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": 0.0,
            "keys": [
                name for name, module in policy.named_modules()
                if isinstance(module, nn.Linear) and name != "lm_head"
            ],
        }
        print("Applying new LoRA adapters to all linear layers ...")
        policy.freeze()
        linear_to_lora_layers(policy, num_layers=-1, config=lora_config)
        adapter_config = {
            "fine_tune_type": "lora",
            "base_model": args.model,
            "num_layers": -1,
            "lora_parameters": lora_config,
        }

    # Full training without an adapter updates every parameter of the full model.
    else:
        policy.unfreeze()

    # Native serialization needs the original policy and adapter configuration.
    policy._checkpoint_config = model_config
    policy._adapter_config = adapter_config

    # GRPO's reference must represent the policy before its first update.
    # Keep dropout disabled for consistent log-probability comparisons; eval mode still allows gradients.
    policy.eval()
    reference = copy.deepcopy(policy)
    reference.freeze()
    reference.eval()

    trainable_parameters = sum(
        parameter.size
        for _, parameter in tree_flatten(policy.trainable_parameters())
    )
    total_parameters = sum(
        parameter.size for _, parameter in tree_flatten(policy.parameters())
    )
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


def generate_rollouts(policy, prompt_tensor, group_size, max_new_tok, tokenizer):
    """Sample a group from the unmodified softmax, padding after generation EOS."""
    # The unpadded prompt is entirely visible, even when PAD and EOS share an ID.
    prompt_ids_extended = mx.repeat(prompt_tensor, group_size, axis=0)  # [G, P]
    cache = make_prompt_cache(policy)
    logits = policy(prompt_ids_extended, cache=cache)[:, -1, :]  # [G, V]
    active = mx.ones((group_size,), dtype=mx.bool_)
    eos_token_ids = mx.array(list(tokenizer.eos_token_ids), dtype=mx.int32)
    completion_steps = []

    for _ in range(max_new_tok):
        # No top-k, top-p, or repetition filtering: match token_logprobs.
        next_token = mx.random.categorical(logits.astype(mx.float32), axis=-1)
        next_token = mx.where(active, next_token, tokenizer.pad_token_id)
        completion_steps.append(next_token)
        is_eos = (next_token[:, None] == eos_token_ids[None, :]).any(axis=-1)
        active = active & ~is_eos
        mx.eval(next_token, active)
        if len(completion_steps) == max_new_tok or not active.any().item():
            break
        logits = policy(next_token[:, None], cache=cache)[:, -1, :]  # [G, V]

    completion_ids = mx.stop_gradient(mx.stack(completion_steps, axis=-1))  # [G, L]

    # Works for Qwen. It produces EOS after generation ends. Fills the rest with PAD.
    is_eos_or_pad = ((completion_ids == tokenizer.eos_token_id) | (completion_ids == tokenizer.pad_token_id))

    # We have to include the first EOS token
    completion_mask = (mx.cumsum(is_eos_or_pad.astype(mx.int32), axis=-1) <= 1).astype(mx.int32)  # [G, L]
    truncated_rollouts = ~is_eos_or_pad.any(axis=-1)  # [G]

    return completion_ids, completion_mask, truncated_rollouts


def token_logprobs(model, prompt_ids, rollouts, rollout_masks):
    """Score completion tokens in float32, excluding masked positions."""
    prompt_ids_extended = mx.repeat(prompt_ids, rollouts.shape[0], axis=0)  # [G, P]
    complete_ids = mx.concatenate([prompt_ids_extended, rollouts], axis=-1)  # [G, P + L]

    # MLX models return full logits; omitting cache keeps scoring differentiable.
    logits = model(complete_ids)

    # Take the completion predictors, excluding the last (out-of-scope) token.
    # Negative indexing also handles models that return the full sequence.
    logits_shift = logits[:, -(rollouts.shape[1] + 1):-1, :]  # [G, L, V]

    # Keep ratios and KL in float32 even when model logits are bfloat16.
    logits_shift = logits_shift.astype(mx.float32)
    logprobs = logits_shift - mx.logsumexp(logits_shift, axis=-1, keepdims=True)  # [G, L, V]

    # Choose probabilities that corresponds to tokens in the rollout with fancy indexing.
    group_indices = mx.arange(rollouts.shape[0])[:, None]
    position_indices = mx.arange(rollouts.shape[1])[None, :]
    selected_logprobs = logprobs[group_indices, position_indices, rollouts]  # [G, L]

    # Apply mask
    selected_logprobs_masked = (
        selected_logprobs * rollout_masks
    )

    return selected_logprobs_masked


def group_advantages(rewards, epsilon):
    """Standardize group rewards, giving constant groups exactly zero advantage."""
    reward_std = rewards.std(ddof=0)
    mixed_group = rewards.max() > rewards.min()
    # Repeated fractional rewards can have a rounded mean unequal to each
    # element. Such constant groups must have exactly zero task advantage.
    return mx.where(
        mixed_group,
        (rewards - rewards.mean()) / (reward_std + epsilon),
        mx.zeros_like(rewards),
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
    """Save native MLX weights, adapters, tokenizer, and update metadata."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:06d}")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_model = policy
    if policy._adapter_config is not None:
        mx.save_safetensors(
            os.path.join(checkpoint_path, "adapters.safetensors"),
            dict(tree_flatten(policy.trainable_parameters())),
        )
        with open(os.path.join(checkpoint_path, "adapter_config.json"), "w") as adapter_file:
            json.dump(policy._adapter_config, adapter_file, indent=2)
        # Fuse a copy so saving never changes the policy or optimizer parameters.
        checkpoint_model = copy.deepcopy(policy)
        checkpoint_model.update_modules(tree_unflatten([
            (name, module.fuse())
            for name, module in checkpoint_model.named_modules()
            if hasattr(module, "fuse")
        ]))
    save_model(checkpoint_path, checkpoint_model)
    save_config(policy._checkpoint_config.copy(), Path(checkpoint_path) / "config.json")
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

    # MLX differentiation selects only trainable parameters (important for LoRA).
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=0.01,
        bias_correction=True,
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
    learning_rate_display = f"{optimizer.learning_rate.item():.3g}".replace("e-0", "e-")
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

        prompt_tensor = mx.array(sample['prompt_ids'], dtype=mx.int32)[None, :]  # [1, P]

        rollouts, rollout_masks, truncated_rollouts = generate_rollouts(
            policy,
            prompt_tensor,
            args.group_size,
            args.max_new_tok,
            tokenizer,
        )  # [G, L], [G, L], [G]

        old_logprobs = mx.stop_gradient(token_logprobs(
            policy,
            prompt_tensor,
            rollouts,
            rollout_masks,
        )) # [G, L]

        reference_logprobs = mx.stop_gradient(token_logprobs(
            reference,
            prompt_tensor,
            rollouts,
            rollout_masks,
        )) # [G, L]
        mx.eval(old_logprobs, reference_logprobs)

        rollouts_text = tokenizer.batch_decode(rollouts.tolist(), skip_special_tokens=True) # G

        if args.dataset == "xlam-tool-calling-mixed":
            reward_values = []
            for rollout_text in rollouts_text:
                _, _, reward_is_one = xlam_function_calling.tool_call_reward(
                    rollout_text,
                    sample["answers"],
                )
                reward_values.append(float(reward_is_one))
            rewards = mx.array(
                reward_values,
                dtype=mx.float32,
            ) # [G]
        else:
            rewards = mx.array(
                gsm8k.answer_rewards(
                    rollouts_text,
                    sample["ground_truth"],
                ),
                dtype=mx.float32,
            ) # [G]

        mixed_group = rewards.max() > rewards.min()
        advantage = group_advantages(rewards, args.epsilon) # [G]
        advantage = advantage[:, None] # [G, 1]

        def grpo_loss(policy):
            """Return the clipped GRPO objective and masked reference KL."""
            new_logprobs = token_logprobs(
                policy,
                prompt_tensor,
                rollouts,
                rollout_masks
            ) # [G, L]

            # Initially, new_logprobs ≈ old_logprobs, so ratio ≈ 1
            ratio = mx.exp(new_logprobs - old_logprobs) # [G, L]

            unclipped = ratio * advantage # [G, L]
            clipped = mx.clip(ratio,
                              1 - args.clip_eps,
                              1 + args.clip_eps) * advantage # [G, L]
            surrogate = mx.minimum(unclipped, clipped) # [G, L]

            logprob_difference = reference_logprobs - new_logprobs # [G, L]

            # expm1 avoids subtracting nearly equal values when the policies agree.
            kl_loss = mx.expm1(logprob_difference) - logprob_difference # [G, L]

            grpo_loss_inside = (-surrogate + args.kl_coef * kl_loss) * rollout_masks # [G, L]

            # rollout_masks.sum() contains total number of tokens (excluding PAD).
            # Guard against a zero denominator for a fully masked sequence.
            grpo_loss = grpo_loss_inside.sum() / mx.maximum(rollout_masks.sum(), 1)
            mean_kl = (kl_loss * rollout_masks).sum() / mx.maximum(rollout_masks.sum(), 1)
            return grpo_loss, mean_kl

        loss_and_grad = nn.value_and_grad(policy, grpo_loss)
        ppo_epoch_losses = []
        ppo_epoch_kl_values = []
        ppo_epoch_gradient_norms = []
        for _ in range(args.ppo_epochs):
            (loss, mean_kl), gradients = loss_and_grad(policy)
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                max_norm=1.0,
            )
            optimizer.update(policy, gradients)
            mx.eval(policy.parameters(), optimizer.state, loss, mean_kl, gradient_norm)

            ppo_epoch_losses.append(mx.stop_gradient(loss))
            ppo_epoch_kl_values.append(mx.stop_gradient(mean_kl))
            ppo_epoch_gradient_norms.append(mx.stop_gradient(gradient_norm))

        completed_step = step + 1
        completed_steps = completed_step
        mean_loss = mx.stack(ppo_epoch_losses).mean().astype(mx.float32).item()
        mean_kl = mx.stack(ppo_epoch_kl_values).mean().astype(mx.float32).item()
        mean_gradient_norm = mx.stack(
            ppo_epoch_gradient_norms
        ).mean().astype(mx.float32).item()
        mean_reward = rewards.mean().item()
        # Both reward variants reserve 1.0 for exact task correctness.
        mean_exact_reward = (rewards == 1.0).astype(mx.float32).mean().item()
        truncated_rollout_fraction = truncated_rollouts.astype(mx.float32).mean().item()
        completion_tokens = rollout_masks.sum().item()
        step_elapsed_time = time.time() - step_start_time

        log_step_count += 1
        loss_sum_since_log += mean_loss
        reward_sum_since_log += mean_reward
        exact_reward_sum_since_log += mean_exact_reward
        truncated_rollout_fraction_sum_since_log += truncated_rollout_fraction
        all_zero_group_count_since_log += int((rewards == 0.0).all().item())
        all_one_group_count_since_log += int((rewards == 1.0).all().item())
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

        learning_rate_display = f"{optimizer.learning_rate.item():.3g}".replace("e-0", "e-")
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
                optimizer.learning_rate.item(),
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
            optimizer.learning_rate.item(),
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
