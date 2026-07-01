#!/usr/bin/env python3
"""
Minimal SFT training script — MLX on Apple Silicon.

Fine-tunes Qwen/Qwen2-0.5B-Instruct-MLX on openai/gsm8k (socratic/train)
for chain-of-thought reasoning using LoRA.

Usage:
    python sft_train_mlx.py

Dependencies:
    pip install mlx mlx-lm datasets transformers tensorboard
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from tensorboardX import SummaryWriter

import data_preperation.gsm8k as _gsm8k_data
import data_preperation.magpie_reasoning as _magpie_data



# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal SFT training script — MLX on Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / data
    parser.add_argument("--model",           type=str,   default="Qwen/Qwen2-0.5B-Instruct-MLX", help="HuggingFace model name or local path")
    parser.add_argument("--dataset",         type=str,   default="gsm8k",                         help="Dataset to use: gsm8k or magpie")
    parser.add_argument("--val-split",        type=float, default=0.1,                              help="Fraction of data held out for validation (0–1)")
    parser.add_argument("--eval-every",       type=int,   default=100,                              help="Evaluate on validation set every N steps. Set to -1 to disable.")
    parser.add_argument("--seed",             type=int,   default=42,                               help="Random seed for data shuffling and train/val split")

    # Training
    parser.add_argument("--max-seq-len",     type=int,   default=512,   help="Maximum sequence length (tokens)")
    parser.add_argument("--batch-size",      type=int,   default=2,     help="Per-step batch size")
    parser.add_argument("--lr",              type=float, default=2e-4,  help="Peak learning rate")
    parser.add_argument("--num-iters",       type=int,   default=500,   help="Total gradient steps")
    parser.add_argument("--warmup-steps",    type=int,   default=20,    help="Linear warmup steps")

    # LoRA
    parser.add_argument("--lora-rank",       type=int,   default=8,     help="LoRA rank (r)")
    parser.add_argument("--lora-alpha",      type=float, default=16.0,  help="LoRA alpha (scale = alpha / rank)")
    parser.add_argument("--lora-layers",     type=int,   default=8,     help="Number of transformer layers to apply LoRA to (last N)")

    # Logging / checkpointing
    parser.add_argument("--log-every",       type=int,   default=10,    help="Print loss + TensorBoard scalars every N steps")
    parser.add_argument("--save-every",      type=int,   default=100,   help="Save adapter checkpoint every N steps")
    parser.add_argument("--checkpoint-dir",  type=str,   default="./checkpoints/sft", help="Directory for checkpoints")
    parser.add_argument("--tensorboard-dir", type=str,   default="./runs/sft",    help="Directory for TensorBoard logs")
    parser.add_argument("--param-log-every", type=int,   default=50,    help="Log LoRA parameter histograms every N steps")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def to_mlx_batch(x, dtype):
    """Convert a dataloader batch item to an MLX array (handles mlx or torch)."""
    if isinstance(x, mx.array):
        return x
    return mx.array(x.numpy(), dtype=dtype)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, val_loader):
    """
    Compute mean SFT loss over the full validation set (no gradients).

    Iterates the DataLoader once and returns a plain Python float.
    """
    total_loss = 0.0
    total_toks = 0

    for input_ids, loss_mask in tqdm(val_loader, desc="  eval", leave=False, unit="batch"):
        input_ids = to_mlx_batch(input_ids, mx.int32)
        loss_mask = to_mlx_batch(loss_mask, mx.float32)
        loss, ntoks = sft_loss(model, input_ids, loss_mask)
        mx.eval(loss, ntoks)

        n = ntoks.item()
        total_loss += loss.item() * n
        total_toks += n

    return total_loss / total_toks if total_toks > 0 else float("nan")


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def sft_loss(model, input_ids, loss_mask):
    """
    Causal LM cross-entropy loss restricted to answer tokens.

    Args:
        model:      MLX nn.Module (language model).
        input_ids:  [B, T]  int32 token ids.
        loss_mask:  [B, T]  float32, 1 = supervised token, 0 = ignored.

    Returns:
        (loss, ntokens)  — both are scalar MLX arrays.
    """
    # Shift: context predicts target one step ahead.
    inputs  = input_ids[:, :-1]   # [B, T-1]  — fed into the model
    targets = input_ids[:, 1:]    # [B, T-1]  — what each position should predict
    mask    = loss_mask[:, 1:]    # [B, T-1]  — aligned with targets

    logits = model(inputs)        # [B, T-1, vocab_size]

    # Per-token cross-entropy, then zero out non-answer positions.
    ce        = nn.losses.cross_entropy(logits, targets, reduction="none")  # [B, T-1]
    masked_ce = ce * mask

    ntoks = mask.sum()
    loss  = masked_ce.sum() / (ntoks + 1e-8)

    return loss, ntoks


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def compute_grad_norm(grads):
    """Compute global L2 norm of all gradients as a scalar MLX array."""
    flat = tree_flatten(grads)
    sq_sum = sum(mx.sum(g * g) for _, g in flat)
    return mx.sqrt(sq_sum)


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_full_checkpoint(model, step, args):
    """
    Save a fully-merged checkpoint loadable by mlx_lm.load().

    Writes adapter weights + adapter_config.json into the checkpoint directory,
    then calls `mlx_lm.fuse` to produce a complete model directory with
    model.safetensors, config.json, and all tokenizer files copied from the
    base model. Adapters are kept on disk for GRPO resume via --load-adapter.

    The live training model is NOT modified — fusing happens inside the
    subprocess on a fresh model load.
    """
    out_dir = os.path.join(args.checkpoint_dir, f"step_{step:06d}")
    os.makedirs(out_dir, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(out_dir, "adapters.safetensors"), adapter_weights)

    adapter_cfg = {
        "fine_tune_type": "lora",
        "base_model": args.model,
        "num_layers": args.lora_layers,
        "lora_parameters": {
            "rank":    args.lora_rank,
            "scale":   args.lora_alpha / args.lora_rank,
            "dropout": 0.05,
        },
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_cfg, f, indent=2)

    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm.fuse",
         "--model",        args.model,
         "--adapter-path", out_dir,
         "--save-path",    out_dir],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"  Warning: mlx_lm.fuse failed (step {step}):")
        print(result.stderr.strip())
    else:
        print(f"  checkpoint -> {out_dir}/  "
              f"(load with: mlx_lm.load('{out_dir}'))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load model --------------------------------------------------------
    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    # Qwen2.5 may not set a pad token; fall back to EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Apply LoRA --------------------------------------------------------
    model.freeze()
    lora_cfg = {
        "rank":    args.lora_rank,
        "scale":   args.lora_alpha / args.lora_rank,  # effective per-layer scale
        "dropout": 0.05,
    }
    linear_to_lora_layers(model, num_layers=args.lora_layers, config=lora_cfg)

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    print(f"LoRA applied: {n_trainable:,} trainable params across {len(trainable)} tensors ({args.lora_layers} layers, r={args.lora_rank})")

    # ---- Load and tokenise data --------------------------------------------
    print("Loading dataset ...")
    if args.dataset == "gsm8k":
        build_dataloaders = _gsm8k_data.build_dataloaders
    elif args.dataset == "magpie":
        build_dataloaders = _magpie_data.build_dataloaders
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: gsm8k, magpie")
    train_loader, val_loader = build_dataloaders(tokenizer, args)

    # ---- Optimizer ---------------------------------------------------------
    # Initialise with peak LR; we override it each step during warmup.
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    # Build the value-and-grad function once (cheaper than rebuilding each step).
    loss_and_grad = nn.value_and_grad(model, sft_loss)

    # ---- TensorBoard -------------------------------------------------------
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs -> {args.tensorboard_dir}  (run: tensorboard --logdir={args.tensorboard_dir})")

    # ---- Training loop -----------------------------------------------------
    print(f"\nTraining: {args.num_iters} steps | batch={args.batch_size} | lr={args.lr} | lora_r={args.lora_rank}\n")

    step     = 0
    tok_s    = 0.0
    t0       = time.time()
    t_log    = time.time()

    data_iter = itertools.chain.from_iterable(
        itertools.repeat(train_loader)
    )

    pbar = tqdm(total=args.num_iters, desc="train", unit="step", dynamic_ncols=True)

    while step < args.num_iters:
        input_ids, loss_mask = next(data_iter)
        input_ids = to_mlx_batch(input_ids, mx.int32)
        loss_mask = to_mlx_batch(loss_mask, mx.float32)

        # Linear warmup: ramp LR from 0 → peak over the first warmup_steps steps.
        if step < args.warmup_steps:
            optimizer.learning_rate = args.lr * (step + 1) / args.warmup_steps
        else:
            optimizer.learning_rate = args.lr

        # Forward pass → loss + gradients in one call.
        (loss, ntoks), grads = loss_and_grad(model, input_ids, loss_mask)

        # Compute gradient norm before the optimizer consumes the gradients.
        gnorm = compute_grad_norm(grads)

        # Optimizer step.
        optimizer.update(model, grads)

        # Flush the lazy computation graph (MLX is lazy by default).
        # Include gnorm so it is evaluated in the same pass.
        mx.eval(model.parameters(), optimizer.state, loss, gnorm)

        # ---- Logging -------------------------------------------------------
        loss_val  = loss.item()
        gnorm_val = gnorm.item()
        lr_val    = float(optimizer.learning_rate)

        # TensorBoard scalars — logged every step.
        writer.add_scalar("train/loss",         loss_val,  step)
        writer.add_scalar("train/learning_rate", lr_val,    step)
        writer.add_scalar("train/grad_norm",     gnorm_val, step)

        if step % args.log_every == 0:
            dt    = time.time() - t_log
            tok_s = ntoks.item() * args.log_every / max(dt, 1e-8) if step > 0 else 0.0
            writer.add_scalar("train/tokens_per_sec", tok_s, step)
            t_log = time.time()

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            lr=f"{lr_val:.2e}",
            gnorm=f"{gnorm_val:.3f}",
            tok_s=f"{tok_s:.0f}",
        )
        pbar.update(1)

        # TensorBoard parameter histograms — logged less frequently.
        if step % args.param_log_every == 0:
            for name, param in tree_flatten(model.trainable_parameters()):
                writer.add_histogram(f"params/{name}", np.array(param), step)

        # ---- Validation ----------------------------------------------------
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader)
            writer.add_scalar("val/loss", val_loss, step)
            pbar.write(f"  [eval] step {step:5d} | val_loss {val_loss:.4f}")

        # ---- Checkpoint ----------------------------------------------------
        if step > 0 and step % args.save_every == 0:
            pbar.write(f"  [ckpt] step {step:5d} | saving checkpoint ...")
            save_full_checkpoint(model, step, args)

        step += 1

    pbar.close()

    # Final checkpoint.
    save_full_checkpoint(model, step, args)
    writer.close()
    print(f"\nDone. Total time: {time.time() - t0:.1f}s")
    print(f"Adapters saved to: {args.checkpoint_dir}/")
    print(f"TensorBoard logs:  tensorboard --logdir={args.tensorboard_dir}")


if __name__ == "__main__":
    main()
