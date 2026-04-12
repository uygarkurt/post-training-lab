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

import json
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from datasets import load_dataset as hf_load_dataset

from tensorboardX import SummaryWriter

# ---------------------------------------------------------------------------
# Config — edit these to change run behaviour
# ---------------------------------------------------------------------------

MODEL_NAME     = "Qwen/Qwen2-0.5B-Instruct-MLX"
DATASET_NAME   = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT  = "train"

MAX_SEQ_LEN    = 512     # Shorter context to fit memory; raise if you have headroom
BATCH_SIZE     = 2       # Per-step batch size
LR             = 2e-4    # Peak learning rate
NUM_ITERS      = 500     # Total gradient steps
WARMUP_STEPS   = 20      # Linear warmup steps

# LoRA
LORA_RANK      = 8       # LoRA rank (r)
LORA_ALPHA     = 16      # LoRA alpha (scale = alpha / rank)
LORA_LAYERS    = 8       # Number of transformer layers to apply LoRA to (last N)

# Logging / checkpointing
LOG_EVERY       = 10      # Print loss + TensorBoard scalars every N steps
SAVE_EVERY      = 100     # Save adapter checkpoint every N steps
CHECKPOINT_DIR  = "./checkpoints"
TENSORBOARD_DIR = "./runs"  # tensorboard --logdir=runs
PARAM_LOG_EVERY = 50      # Log LoRA parameter histograms every N steps

# ---------------------------------------------------------------------------
# Data loading and tokenisation
# ---------------------------------------------------------------------------

def load_and_tokenise(tokenizer):
    """
    Load GSM8K and produce a list of (input_ids, loss_mask) pairs.

    loss_mask is a binary float list:
        0 = prompt token  (no gradient contribution)
        1 = answer token  (supervised)
    """
    ds = hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

    samples = []
    skipped = 0

    for row in ds:
        question = row["question"]
        answer   = row["answer"]

        # Render the user prompt so we can find where the answer begins.
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full conversation: user question + assistant CoT answer.
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer.encode(prompt_text)
        full_ids   = tokenizer.encode(full_text)

        # Skip degenerate samples.
        if len(full_ids) < 4 or len(prompt_ids) >= len(full_ids):
            skipped += 1
            continue

        # Truncate to MAX_SEQ_LEN.
        full_ids = full_ids[:MAX_SEQ_LEN]

        # Build binary loss mask aligned with full_ids.
        # Everything up to (and including) the prompt gets 0; the rest gets 1.
        prompt_len = min(len(prompt_ids), len(full_ids))
        loss_mask  = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
        loss_mask  = loss_mask[:MAX_SEQ_LEN]

        # Skip if the answer was entirely cropped away.
        if sum(loss_mask) == 0:
            skipped += 1
            continue

        samples.append((full_ids, loss_mask))

    print(f"  {len(samples)} samples loaded, {skipped} skipped.")
    return samples


def iterate_batches(samples, batch_size, pad_id):
    """
    Infinitely yields shuffled batches of (input_ids, loss_mask) as MLX arrays.

    Sequences within a batch are right-padded to the same length.
    Padding positions are masked out (loss_mask = 0) automatically.
    """
    while True:
        indices = list(range(len(samples)))
        random.shuffle(indices)

        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_idx   = indices[start : start + batch_size]
            batch_ids   = [samples[i][0] for i in batch_idx]
            batch_masks = [samples[i][1] for i in batch_idx]

            max_len = max(len(ids) for ids in batch_ids)

            padded_ids   = []
            padded_masks = []
            for ids, mask in zip(batch_ids, batch_masks):
                pad = max_len - len(ids)
                padded_ids.append(ids   + [pad_id] * pad)
                padded_masks.append(mask + [0]     * pad)

            yield (
                mx.array(padded_ids,   dtype=mx.int32),
                mx.array(padded_masks, dtype=mx.float32),
            )


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

def save_full_checkpoint(model, step):
    """
    Save a fully-merged checkpoint loadable by mlx_lm.load().

    Writes adapter weights + adapter_config.json to a temp directory, then
    calls `mlx_lm.fuse` (same approach as mlx-tune's SFTTrainer) to produce
    a complete model directory with model.safetensors, config.json, and all
    tokenizer files copied from the base model.

    The live training model is NOT modified — fusing happens inside the
    subprocess on a fresh model load.
    """
    out_dir = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Dump current LoRA adapter weights.
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(os.path.join(tmp, "adapters.safetensors"), adapter_weights)

        # 2. Write adapter_config.json — required by mlx_lm.fuse.
        #    num_layers must be a top-level key; mlx_lm reads it as
        #    config.num_layers (SimpleNamespace) before applying LoRA layers.
        adapter_cfg = {
            "fine_tune_type": "lora",
            "num_layers": LORA_LAYERS,
            "lora_parameters": {
                "rank":    LORA_RANK,
                "scale":   LORA_ALPHA / LORA_RANK,
                "dropout": 0.05,
            },
        }
        with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
            json.dump(adapter_cfg, f, indent=2)

        # 3. Fuse adapters into the base model and write the full checkpoint.
        #    mlx_lm.fuse copies config.json, tokenizer.json, etc. from the
        #    base model cache automatically.
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm.fuse",
             "--model",        MODEL_NAME,
             "--adapter-path", tmp,
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
    # ---- Load model --------------------------------------------------------
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load(MODEL_NAME)

    # Qwen2.5 may not set a pad token; fall back to EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Apply LoRA --------------------------------------------------------
    model.freeze()
    lora_cfg = {
        "rank":    LORA_RANK,
        "scale":   LORA_ALPHA / LORA_RANK,  # effective per-layer scale
        "dropout": 0.05,
    }
    linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=lora_cfg)

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    print(f"LoRA applied: {n_trainable:,} trainable params across {len(trainable)} tensors")

    # ---- Load and tokenise data --------------------------------------------
    print("Loading dataset ...")
    samples = load_and_tokenise(tokenizer)

    # ---- Optimizer ---------------------------------------------------------
    # Initialise with peak LR; we override it each step during warmup.
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)

    # Build the value-and-grad function once (cheaper than rebuilding each step).
    loss_and_grad = nn.value_and_grad(model, sft_loss)

    # ---- TensorBoard -------------------------------------------------------
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    print(f"TensorBoard logs -> {TENSORBOARD_DIR}  (run: tensorboard --logdir={TENSORBOARD_DIR})")

    # ---- Training loop -----------------------------------------------------
    print(f"\nTraining: {NUM_ITERS} steps | batch={BATCH_SIZE} | lr={LR} | lora_r={LORA_RANK}\n")

    step     = 0
    ema_loss = None
    ema_beta = 0.9
    t0       = time.time()
    t_log    = time.time()

    data_iter = iterate_batches(samples, BATCH_SIZE, tokenizer.pad_token_id)

    while step < NUM_ITERS:
        input_ids, loss_mask = next(data_iter)

        # Linear warmup: ramp LR from 0 → peak over the first WARMUP_STEPS steps.
        if step < WARMUP_STEPS:
            optimizer.learning_rate = LR * (step + 1) / WARMUP_STEPS
        else:
            optimizer.learning_rate = LR

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
        ema_loss  = loss_val if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * loss_val

        # TensorBoard scalars — logged every step.
        writer.add_scalar("train/loss",         loss_val,  step)
        writer.add_scalar("train/loss_ema",      ema_loss,  step)
        writer.add_scalar("train/learning_rate", lr_val,    step)
        writer.add_scalar("train/grad_norm",     gnorm_val, step)

        if step % LOG_EVERY == 0:
            dt    = time.time() - t_log
            tok_s = ntoks.item() * LOG_EVERY / max(dt, 1e-8) if step > 0 else 0.0
            writer.add_scalar("train/tokens_per_sec", tok_s, step)
            print(
                f"step {step:5d} | "
                f"loss {ema_loss:.4f} | "
                f"lr {lr_val:.2e} | "
                f"gnorm {gnorm_val:.3f} | "
                f"tok/s {tok_s:6.0f} | "
                f"elapsed {time.time() - t0:5.0f}s"
            )
            t_log = time.time()

        # TensorBoard parameter histograms — logged less frequently.
        if step % PARAM_LOG_EVERY == 0:
            for name, param in tree_flatten(model.trainable_parameters()):
                writer.add_histogram(f"params/{name}", np.array(param), step)

        # ---- Checkpoint ----------------------------------------------------
        if step > 0 and step % SAVE_EVERY == 0:
            save_full_checkpoint(model, step)

        step += 1

    # Final checkpoint.
    save_full_checkpoint(model, step)
    writer.close()
    print(f"\nDone. Total time: {time.time() - t0:.1f}s")
    print(f"Adapters saved to: {CHECKPOINT_DIR}/")
    print(f"TensorBoard logs:  tensorboard --logdir={TENSORBOARD_DIR}")


if __name__ == "__main__":
    main()
