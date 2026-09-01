import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data_preparation import gsm8k


def parse_args():
    """Parse the small set of options needed by the SFT scaffold."""
    parser = argparse.ArgumentParser(
        description="SFT for MLX (Apple Silicon).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Overfit a tiny GSM8K subset (same samples for train and val)",
    )
    parser.add_argument(
        "--debug-samples", type=int, default=8,
        help="Number of GSM8K samples in --debug mode (train and val use the same set)",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2-0.5B-Instruct-MLX",
        help="Hugging Face model or local model directory",
    )

    parser.add_argument("--batch-size", type=int, default=2, help="Per-step batch size")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum tokenized sample length",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate")
    parser.add_argument("--num-iters", type=int, default=500, help="Number of optimizer steps")

    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale = alpha / rank)")
    parser.add_argument(
        "--lora-layers", type=int, default=-1,
        help="Number of final transformer layers using LoRA (-1 for all layers)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization and data shuffling")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from GSM8K train set")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Skip GSM8K prompts longer than this")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")

    parser.add_argument("--tensorboard-dir", type=str, default="./runs/mlx/sft", help="Base path for timestamped TensorBoard run directories")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/mlx/sft", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.debug_samples < 1:
        parser.error("--debug-samples must be at least 1")
    if args.num_iters < 1:
        parser.error("--num-iters must be at least 1")
    if args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2")
    if args.max_prompt_len < 1:
        parser.error("--max-prompt-len must be at least 1")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be at least 1")
    if args.lora_layers == 0 or args.lora_layers < -1:
        parser.error("--lora-layers must be -1 or at least 1")
    return args


def set_random_seed(seed):
    """Seed Python, NumPy, and MLX for repeatable execution."""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def to_mlx_batch(batch_item, dtype):
    """Convert a PyTorch DataLoader item to an MLX array."""
    if isinstance(batch_item, mx.array):
        return batch_item.astype(dtype)
    return mx.array(batch_item.numpy(), dtype=dtype)


def sft_loss(model, input_ids, loss_mask):
    """Return response-token SFT loss and the supervised token count."""
    logits = model(input_ids)  # [B, L, V]

    logits_shifted = logits[:, :-1, :]  # [B, L-1, V]
    targets = input_ids[:, 1:]  # [B, L-1]
    loss_mask_shifted = loss_mask[:, 1:]  # [B, L-1]

    cross_entropy = nn.losses.cross_entropy(
        logits_shifted,
        targets,
        reduction="none",
    )  # [B, L-1]

    masked_cross_entropy = cross_entropy * loss_mask_shifted
    response_token_count = loss_mask_shifted.sum()
    loss = masked_cross_entropy.sum() / response_token_count
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
        input_ids = to_mlx_batch(input_ids, mx.int32)  # [B, L]
        loss_mask = to_mlx_batch(loss_mask, mx.float32)  # [B, L]
        loss, response_token_count = sft_loss(model, input_ids, loss_mask)
        mx.eval(loss, response_token_count)

        response_token_count_value = int(response_token_count.item())
        total_loss += loss.item() * response_token_count_value
        total_response_token_count += response_token_count_value

    if was_training:
        model.train()

    if total_response_token_count == 0:
        return float("nan")
    return total_loss / total_response_token_count


def save_checkpoint(model, checkpoint_dir, step, args):
    """Save MLX adapter weights and a fused model for one step."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:06d}")
    os.makedirs(checkpoint_path, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(
        os.path.join(checkpoint_path, "adapters.safetensors"),
        adapter_weights,
    )

    adapter_config = {
        "fine_tune_type": "lora",
        "base_model": args.model,
        "num_layers": args.lora_layers,
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": 0.0,
        },
    }
    with open(
        os.path.join(checkpoint_path, "adapter_config.json"),
        "w",
    ) as adapter_config_file:
        json.dump(adapter_config, adapter_config_file, indent=2)

    fuse_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            args.model,
            "--adapter-path",
            checkpoint_path,
            "--save-path",
            checkpoint_path,
        ],
        capture_output=True,
        text=True,
    )

    if fuse_result.returncode != 0:
        print(f"  Warning: mlx_lm.fuse failed at step {step}:")
        print(fuse_result.stderr.strip())
    print(f"  [ckpt] step {step:5d} -> {checkpoint_path}")


def main():
    """Set up SFT infrastructure and run the intentionally minimal loop."""
    args = parse_args()
    saved_args = vars(args).copy()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"sft_{timestamp}"
    args.tensorboard_dir = os.path.join(
        os.path.normpath(args.tensorboard_dir),
        run_name,
    )
    args.checkpoint_dir = os.path.join(
        os.path.normpath(args.checkpoint_dir),
        run_name,
    )
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

    print(f"Loading model {args.model} ...")
    model, tokenizer = load(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.lora_layers > len(model.layers):
        raise ValueError(
            f"Requested LoRA for {args.lora_layers} layers, but the model "
            f"only has {len(model.layers)} layers."
        )

    model.freeze()
    lora_config = {
        "rank": args.lora_rank,
        "scale": args.lora_alpha / args.lora_rank,
        "dropout": 0.0,
    }
    linear_to_lora_layers(
        model,
        num_layers=args.lora_layers,
        config=lora_config,
    )

    trainable_parameters = list(tree_flatten(model.trainable_parameters()))
    trainable_parameter_count = sum(
        parameter.size
        for _, parameter in trainable_parameters
    )
    total_parameter_count = sum(
        parameter.size
        for _, parameter in tree_flatten(model.parameters())
    )
    print(
        "Training mode: lora | "
        f"{trainable_parameter_count:,} / {total_parameter_count:,} "
        "parameters trainable"
    )

    print("Loading GSM8K dataset ...")
    if args.debug:
        train_dataset, val_dataset = (
            gsm8k.GSM8KSFTDataset.build_debug_overfit_datasets(
                tokenizer,
                max_seq_len=args.max_seq_len,
                max_prompt_len=args.max_prompt_len,
                seed=args.seed,
                debug_samples=args.debug_samples,
            )
        )
    else:
        train_dataset, val_dataset = (
            gsm8k.GSM8KSFTDataset.build_train_val_datasets(
                tokenizer,
                max_seq_len=args.max_seq_len,
                max_prompt_len=args.max_prompt_len,
                val_split=args.val_split,
                seed=args.seed,
            )
        )
    train_loader = gsm8k.build_sft_dataloader(
        train_dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = gsm8k.build_sft_dataloader(
        val_dataset,
        tokenizer,
        batch_size=args.batch_size,
    )

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=0.01,
        bias_correction=True,
    )
    loss_and_grad = nn.value_and_grad(model, sft_loss)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs: tensorboard --logdir={args.tensorboard_dir}")

    if args.eval_every != -1:
        validation_loss = calculate_validation_loss(model, val_loader)
        writer.add_scalar("val/loss", validation_loss, 0)
        print(f"  [val] step {0:5d} | loss {validation_loss:.4f}")

    model.train()
    progress = tqdm(
        total=args.num_iters,
        desc="train loss=----",
        unit="step",
        ncols=terminal_columns,
    )
    completed_steps = 0

    while completed_steps < args.num_iters:
        for input_ids, loss_mask in train_loader:
            if completed_steps >= args.num_iters:
                break

            step = completed_steps + 1
            step_start_time = time.time()
            input_ids = to_mlx_batch(input_ids, mx.int32)  # [B, L]
            loss_mask = to_mlx_batch(loss_mask, mx.float32)  # [B, L]

            (loss, response_token_count), gradients = loss_and_grad(
                model,
                input_ids,
                loss_mask,
            )
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                max_norm=1.0,
            )
            optimizer.update(model, gradients)
            mx.eval(
                model.parameters(),
                optimizer.state,
                loss,
                gradient_norm,
                response_token_count,
            )

            loss_value = loss.item()
            gradient_norm_value = gradient_norm.item()
            response_token_count_value = int(response_token_count.item())
            tokens_per_second = response_token_count_value / max(
                time.time() - step_start_time,
                1e-8,
            )
            completed_steps = step

            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/grad_norm", gradient_norm_value, step)
            writer.add_scalar("train/learning_rate", args.lr, step)
            writer.add_scalar(
                "train/response_tokens",
                response_token_count_value,
                step,
            )
            writer.add_scalar("train/tokens_per_sec", tokens_per_second, step)

            progress.set_description(f"train loss={loss_value:.4f}")
            progress.update(1)

            if args.eval_every > 0 and step % args.eval_every == 0:
                validation_loss = calculate_validation_loss(model, val_loader)
                writer.add_scalar("val/loss", validation_loss, step)
                progress.write(
                    f"  [val] step {step:5d} | loss {validation_loss:.4f}"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                writer.flush()
                save_checkpoint(model, args.checkpoint_dir, step, args)

    progress.close()

    if (
        args.save_every > 0
        and completed_steps > 0
        and completed_steps % args.save_every != 0
    ):
        writer.flush()
        save_checkpoint(model, args.checkpoint_dir, completed_steps, args)

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
