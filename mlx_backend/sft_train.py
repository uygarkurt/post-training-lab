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
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
from mlx_lm.utils import save_config, save_model
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_preparation import gsm8k, sft, xlam_function_calling


def parse_args():
    """Parse the small set of options needed by the SFT scaffold."""
    parser = argparse.ArgumentParser(
        description="SFT for MLX (Apple Silicon).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Overfit a tiny subset (same samples for train and val)",
    )
    parser.add_argument(
        "--debug-samples", type=int, default=8,
        help="Number of samples in --debug mode (train and val use the same set)",
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "xlam-tool-calling-mixed", "gsm8k",
        ),
        default="gsm8k",
        help="SFT dataset to train on",
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

    parser.add_argument("--batch-size", type=int, default=2, help="Per-step batch size")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum tokenized sample length",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak AdamW learning rate for cosine scheduling")
    parser.add_argument("--num-iters", type=int, default=500, help="Number of optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Steps to linearly increase the learning rate to --lr (0 disables warmup)")
    parser.add_argument("--min-lr", type=float, default=5e-6, help="Minimum learning rate at the end of the cosine schedule")

    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale = alpha / rank)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization and data shuffling")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from the training dataset")
    parser.add_argument("--log-every", type=int, default=10, help="Write training metrics to TensorBoard every N steps (loss averaged over response tokens)")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/cuda/sft", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    # The scheduler accepts some combinations that silently change its behavior.
    if args.num_iters < 1:
        parser.error("--num-iters must be at least 1")
    if not 0 <= args.warmup_steps < args.num_iters:
        parser.error("--warmup-steps must be at least 0 and less than --num-iters")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and greater than 0")
    if not 0 <= args.min_lr <= args.lr:
        parser.error("--min-lr must be between 0 and --lr")
    return args


def set_random_seed(seed):
    """Seed Python, NumPy, PyTorch data loading, and MLX."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)


def load_model_and_tokenizer(args):
    """Load a dense MLX model or a native MLX adapter for training."""
    print(f"Loading model {args.model} ...")
    model, tokenizer, model_config = load(args.model, return_config=True)
    model.set_dtype(mx.bfloat16)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    adapter_config = None
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
        model.freeze()
        load_adapters(model, adapter_path)

        if args.train_mode == "full":
            print("Merging adapter into the model for full training ...")
            model.update_modules(tree_unflatten([
                (name, module.fuse())
                for name, module in model.named_modules()
                if hasattr(module, "fuse")
            ]))
            model.unfreeze()
            adapter_config = None

    elif args.train_mode == "lora":
        # Explicit keys match PEFT's all-linear targets, excluding the output head.
        lora_config = {
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": 0.0,
            "keys": [
                name for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name != "lm_head"
            ],
        }
        print("Applying new LoRA adapters to all linear layers ...")
        model.freeze()
        linear_to_lora_layers(model, num_layers=-1, config=lora_config)
        adapter_config = {
            "fine_tune_type": "lora",
            "base_model": args.model,
            "num_layers": -1,
            "lora_parameters": lora_config,
        }

    else:
        model.unfreeze()

    # Native serialization needs the original model and adapter configuration.
    model._checkpoint_config = model_config
    model._adapter_config = adapter_config

    trainable_parameters = sum(
        parameter.size
        for _, parameter in tree_flatten(model.trainable_parameters())
    )
    total_parameters = sum(
        parameter.size for _, parameter in tree_flatten(model.parameters())
    )
    print(
        f"Training mode: {args.train_mode} | "
        f"{trainable_parameters:,} / {total_parameters:,} parameters trainable"
    )
    return model, tokenizer


def load_sft_datasets(tokenizer, args):
    """Load the selected SFT datasets and return their matching collator builder."""
    if args.dataset == "xlam-tool-calling-mixed":
        print("Loading xlam-tool-calling-mixed dataset ...")
        dataset_class = xlam_function_calling.XLAMFunctionCallingSFTDataset
        build_dataloader = sft.build_sft_dataloader
    else:
        print("Loading GSM8K dataset ...")
        dataset_class = gsm8k.GSM8KSFTDataset
        build_dataloader = gsm8k.build_sft_dataloader

    if args.debug:
        train_dataset, val_dataset = dataset_class.build_debug_overfit_datasets(
            tokenizer,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            debug_samples=args.debug_samples,
        )
    else:
        train_dataset, val_dataset = dataset_class.build_train_val_datasets(
            tokenizer,
            max_seq_len=args.max_seq_len,
            val_split=args.val_split,
            seed=args.seed,
        )

    return train_dataset, val_dataset, build_dataloader


def sft_loss(model, input_ids, loss_mask):
    """Return response-token SFT loss and the supervised token count."""
    logits = model(input_ids)  # [B, L, V]

    logits_shifted = logits[:, :-1, :]  # [B, L-1, V]
    targets = input_ids[:, 1:]  # [B, L-1]
    loss_mask_shifted = loss_mask[:, 1:]  # [B, L-1]

    ce = nn.losses.cross_entropy(
        logits_shifted.astype(mx.float32),
        targets,
        reduction="none",
    ).astype(logits_shifted.dtype)  # [B, L-1]; accumulate softmax in float32

    masked_ce = ce * loss_mask_shifted  # [B, L-1]
    response_token_count = loss_mask_shifted.sum()
    loss = masked_ce.sum() / response_token_count
    return loss, response_token_count


def calculate_validation_loss(model, val_loader):
    """Return response-token-weighted loss over the validation loader."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_response_token_count = 0

    for input_ids, loss_mask in tqdm(
        val_loader,
        desc="eval",
        leave=False,
        unit="batch",
    ):
        input_ids = mx.array(input_ids.numpy(), dtype=mx.int32)  # [B, L]
        loss_mask = mx.array(loss_mask.numpy(), dtype=mx.float32)  # [B, L]

        logits = model(input_ids)  # [B, L, V]

        logits_shifted = logits[:, :-1, :]  # [B, L-1, V]
        targets = input_ids[:, 1:]  # [B, L-1]
        loss_mask_shifted = loss_mask[:, 1:]  # [B, L-1]

        ce = nn.losses.cross_entropy(
            logits_shifted.astype(mx.float32),
            targets,
            reduction="none",
        ).astype(logits_shifted.dtype)  # [B, L-1]; accumulate softmax in float32

        masked_ce = ce * loss_mask_shifted
        response_token_count = int(loss_mask_shifted.sum().item())

        total_loss += masked_ce.astype(mx.float32).sum().item()
        total_response_token_count += response_token_count

    if was_training:
        model.train()

    if total_response_token_count == 0:
        return float("nan")
    return total_loss / total_response_token_count


def save_checkpoint(
    model, tokenizer, checkpoint_dir, step, training_start_time, learning_rate,
):
    """Save native MLX weights, adapters, tokenizer, and update metadata."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:06d}")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_model = model
    if model._adapter_config is not None:
        mx.save_safetensors(
            os.path.join(checkpoint_path, "adapters.safetensors"),
            dict(tree_flatten(model.trainable_parameters())),
        )
        with open(os.path.join(checkpoint_path, "adapter_config.json"), "w") as adapter_file:
            json.dump(model._adapter_config, adapter_file, indent=2)
        # Fuse a copy so saving never changes the policy or optimizer parameters.
        checkpoint_model = copy.deepcopy(model)
        checkpoint_model.update_modules(tree_unflatten([
            (name, module.fuse())
            for name, module in checkpoint_model.named_modules()
            if hasattr(module, "fuse")
        ]))
    save_model(checkpoint_path, checkpoint_model)
    save_config(model._checkpoint_config.copy(), Path(checkpoint_path) / "config.json")
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
    """Set up SFT infrastructure and run the intentionally minimal loop."""
    args = parse_args()
    saved_args = vars(args).copy()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"sft_{timestamp}"
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

    set_random_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset, val_dataset, build_dataloader = load_sft_datasets(
        tokenizer,
        args,
    )
    train_loader = build_dataloader(
        train_dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        tokenizer,
        batch_size=args.batch_size,
    )

    scheduler = optim.cosine_decay(
        args.lr,
        args.num_iters - args.warmup_steps,
        end=args.min_lr,
    )
    if args.warmup_steps:
        scheduler = optim.join_schedules(
            [optim.linear_schedule(0.0, args.lr, args.warmup_steps), scheduler],
            [args.warmup_steps],
        )
    optimizer = optim.AdamW(
        learning_rate=scheduler(0),
        weight_decay=0.01,
        bias_correction=True,
    )
    loss_and_grad = nn.value_and_grad(model, sft_loss)
    writer = SummaryWriter(log_dir=run_logs_dir)
    print(f"TensorBoard logs: tensorboard --logdir={run_logs_dir}")

    if args.eval_every != -1:
        validation_loss = calculate_validation_loss(model, val_loader)
        writer.add_scalar("val/loss", validation_loss, 0)
        print(f"  [val] step {0:5d} | loss {validation_loss:.4f}")

    model.train()
    batches_per_epoch = len(train_loader)
    learning_rate_display = f"{optimizer.learning_rate.item():.3g}".replace("e-0", "e-")
    progress = tqdm(
        total=args.num_iters,
        desc=(
            f"epoch=1 batch=0/{batches_per_epoch} loss=---- "
            f"lr={learning_rate_display}"
        ),
        unit="step",
        ncols=terminal_columns,
    )
    completed_steps = 0
    epoch_number = 0
    examples_seen = 0
    loss_sum_since_log = 0.0
    response_tokens_since_log = 0
    training_start_time = time.monotonic()

    while completed_steps < args.num_iters:
        epoch_number += 1
        for batch_index, (input_ids, loss_mask) in enumerate(train_loader, start=1):
            if completed_steps >= args.num_iters:
                break

            step = completed_steps + 1
            step_start_time = time.time()
            input_ids = mx.array(input_ids.numpy(), dtype=mx.int32)  # [B, L]
            loss_mask = mx.array(loss_mask.numpy(), dtype=mx.float32)  # [B, L]

            (loss, response_token_count), gradients = loss_and_grad(
                model, input_ids, loss_mask,
            )
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                max_norm=1.0,
            )

            # Record the rate used for this update before advancing the schedule.
            learning_rate = optimizer.learning_rate.item()
            optimizer.update(model, gradients)
            optimizer.learning_rate = scheduler(step)
            mx.eval(model.parameters(), optimizer.state, loss, gradient_norm, response_token_count)

            loss_value = loss.astype(mx.float32).item()
            gradient_norm_value = gradient_norm.astype(mx.float32).item()
            response_token_count = int(response_token_count.item())
            tokens_per_second = response_token_count / max(
                time.time() - step_start_time,
                1e-8,
            )
            completed_steps = step
            examples_seen += input_ids.shape[0]
            dataset_passes = examples_seen / len(train_dataset)

            loss_sum_since_log += loss_value * response_token_count
            response_tokens_since_log += response_token_count
            if step % args.log_every == 0 or step == args.num_iters:
                writer.add_scalar(
                    "train/loss",
                    loss_sum_since_log / response_tokens_since_log,
                    step,
                )
                writer.add_scalar("train/grad_norm", gradient_norm_value, step)
                writer.add_scalar("train/learning_rate", learning_rate, step)
                writer.add_scalar("train/response_tokens", response_token_count, step)
                writer.add_scalar("train/tokens_per_sec", tokens_per_second, step)
                writer.add_scalar("train/epoch", dataset_passes, step)
                writer.add_scalar("train/examples_seen", examples_seen, step)
                loss_sum_since_log = 0.0
                response_tokens_since_log = 0

            learning_rate_display = f"{learning_rate:.3g}".replace("e-0", "e-")
            progress.set_description(
                f"epoch={epoch_number} batch={batch_index}/{batches_per_epoch} "
                f"loss={loss_value:.4f} lr={learning_rate_display}",
                refresh=False,
            )
            progress.update(1)

            if examples_seen % len(train_dataset) == 0:
                progress.write(
                    f"  [data] completed dataset pass {int(dataset_passes)} | "
                    f"step {step:5d}"
                )

            if args.eval_every > 0 and step % args.eval_every == 0:
                validation_loss = calculate_validation_loss(model, val_loader)
                writer.add_scalar("val/loss", validation_loss, step)
                progress.write(
                    f"  [val] step {step:5d} | loss {validation_loss:.4f}"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                writer.flush()
                save_checkpoint(
                    model, tokenizer, args.checkpoint_dir, step, training_start_time,
                    learning_rate,
                )

    progress.close()

    if (
        args.save_every > 0
        and completed_steps > 0
        and completed_steps % args.save_every != 0
    ):
        writer.flush()
        save_checkpoint(
            model, tokenizer, args.checkpoint_dir, completed_steps, training_start_time,
            learning_rate,
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
