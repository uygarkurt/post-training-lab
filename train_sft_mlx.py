#!/usr/bin/env python3
"""
Optimized SFT Script for Training Reasoning Models on Apple Silicon
Uses MLX framework for M1/M2/M3 chips
"""

import argparse
import json
import logging
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from mlx_lm import load
from safetensors.numpy import save_file as safetensors_save

from data_prep import get_prep_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")


class MLXDataLoader:
    """Simple data loader for MLX that yields batches."""
    
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            yield self.dataset[idx]


def prepare_datasets(args, tokenizer):
    """Load and split dataset using dataset-specific preparation module."""
    logging.info(f"Loading dataset: {args.dataset_name}")
    
    # Auto-select the preparation module based on dataset name
    
    prep_module = get_prep_module(args.dataset_name)
    
    # Call the prepare_data function from the module
    train_dataset, val_dataset = prep_module.prepare_data(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        validation_split=args.validation_split,
    )
    
    logging.info(f"Train: {len(train_dataset)} • Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def create_optimizer_and_scheduler(model, args, total_steps: int):
    """Create AdamW optimizer with cosine warmup scheduler."""
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return args.learning_rate * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.learning_rate * max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    logging.info(f"AdamW optimizer • Cosine scheduler • {warmup_steps} warmup steps")
    
    return optimizer, lr_schedule


def save_checkpoint(model, tokenizer, model_config, epoch, step, args):
    """Save checkpoint and manage limits."""
    import shutil
    
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save MLX model weights
    save_model(model, tokenizer, model_config, ckpt_dir)
    
    # Save training state
    state = {
        "epoch": epoch,
        "step": step,
        "config": vars(args)
    }
    
    with open(ckpt_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    # Keep only last N checkpoints
    checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"), 
                        key=lambda x: int(x.name.split("-")[1]))
    for old_ckpt in checkpoints[:-args.save_total_limit]:
        shutil.rmtree(old_ckpt)
    
    logging.info(f"Saved checkpoint-{step}")


def save_model(model, tokenizer, model_config, output_dir: Path):
    """Save MLX model as safetensors format compatible with mlx_lm.load()."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert MLX weights to numpy and save as safetensors
    weights_dict = {}
    for key, value in tree_flatten(model.parameters()):
        # Convert MLX array to numpy
        weights_dict[key] = np.array(value)
    
    # Save as safetensors
    safetensors_save(weights_dict, str(output_dir / "model.safetensors"))
    
    # Save tokenizer files
    tokenizer.save_pretrained(output_dir)
    
    # Save model config (architecture)
    model_config.save_pretrained(output_dir)
    
    logging.info(f"Model saved to {output_dir}")


def evaluate(model, dataloader, args):
    """Run validation and return average loss."""
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        # Convert to MLX arrays and add batch dimension
        input_ids = mx.array(batch["input_ids"].numpy())[None, :]  # Add batch dim
        labels = mx.array(batch["labels"].numpy())[None, :]  # Add batch dim
        
        # Forward pass
        logits = model(input_ids)
        
        # Remove batch dimension from logits
        logits = logits[0]  # Shape: (seq_len, vocab_size)
        labels_unbatched = labels[0]  # Shape: (seq_len,)
        
        # Compute loss for all positions
        losses = nn.losses.cross_entropy(logits, labels_unbatched, reduction="none")
        
        # Mask out padding tokens
        mask = (labels_unbatched != -100).astype(mx.float32)
        num_valid = mx.sum(mask)
        
        if num_valid > 0:
            masked_losses = losses * mask
            loss = mx.sum(masked_losses) / num_valid
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def loss_fn(model, input_ids, labels):
    """Compute loss for training."""
    logits = model(input_ids)
    
    # Remove batch dimension
    logits = logits[0]  # Shape: (seq_len, vocab_size)
    labels = labels[0]  # Shape: (seq_len,)
    
    # Compute loss for all positions
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    
    # Mask out padding tokens (labels == -100)
    mask = (labels != -100).astype(mx.float32)
    
    # Compute mean loss only over non-masked tokens
    masked_losses = losses * mask
    num_valid = mx.sum(mask)
    
    if num_valid == 0:
        return mx.array(0.0)
    
    loss = mx.sum(masked_losses) / num_valid
    return loss


def train(model, tokenizer, model_config, train_loader, val_loader, optimizer, lr_schedule, args):
    """Main training loop with gradient accumulation."""
    global_step = 0
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    
    logging.info("=" * 80)
    logging.info(f"Training on Apple Silicon • Epochs: {args.num_epochs}")
    logging.info(f"Actual batch: {args.batch_size} • Grad accumulation: {args.gradient_accumulation_steps} • Effective batch: {effective_batch}")
    logging.info("=" * 80)
    
    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    for epoch in range(args.num_epochs):
        accumulated_grads = None
        num_accumulated = 0
        epoch_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress):
            # Convert to MLX arrays and add batch dimension
            input_ids = mx.array(batch["input_ids"].numpy())[None, :]  # Shape: (1, seq_len)
            labels = mx.array(batch["labels"].numpy())[None, :]  # Shape: (1, seq_len)
            
            # Forward and backward pass
            loss, grads = loss_and_grad_fn(model, input_ids, labels)
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map_with_two(
                    lambda a, b: a + b, accumulated_grads, grads
                )
            num_accumulated += 1
            epoch_loss += loss.item()
            
            # Update weights after gradient accumulation
            if num_accumulated >= args.gradient_accumulation_steps:
                # Average accumulated gradients
                avg_grads = tree_map(
                    lambda g: g / num_accumulated, accumulated_grads
                )
                
                # Gradient clipping
                grad_norm = tree_reduce(
                    lambda acc, g: acc + mx.sum(g * g), 
                    avg_grads, 
                    mx.array(0.0)
                )
                grad_norm = mx.sqrt(grad_norm)
                
                if grad_norm > args.max_grad_norm:
                    scale = args.max_grad_norm / grad_norm
                    avg_grads = tree_map(lambda g: g * scale, avg_grads)
                
                # Update learning rate
                current_lr = lr_schedule(global_step)
                optimizer.learning_rate = current_lr
                
                # Optimizer step
                optimizer.update(model, avg_grads)
                mx.eval(model.parameters())
                
                # Reset accumulation
                accumulated_grads = None
                num_accumulated = 0
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    progress.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step
                    })
                
                # Validation
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, args)
                    logging.info(f"Step {global_step} • Val loss: {val_loss:.4f}")
                
                # Checkpointing
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, tokenizer, model_config, epoch, global_step, args)
                
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        
        save_checkpoint(model, tokenizer, model_config, epoch + 1, global_step, args)
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    logging.info("Training completed!")
    return model


def tree_map(fn, tree):
    """Apply function to all arrays in a tree."""
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, x) for x in tree)
    else:
        return fn(tree)


def tree_map_with_two(fn, tree1, tree2):
    """Apply binary function to two trees element-wise."""
    if isinstance(tree1, dict):
        return {k: tree_map_with_two(fn, tree1[k], tree2[k]) for k in tree1.keys()}
    elif isinstance(tree1, (list, tuple)):
        return type(tree1)(tree_map_with_two(fn, a, b) for a, b in zip(tree1, tree2))
    else:
        return fn(tree1, tree2)


def tree_reduce(fn, tree, init):
    """Reduce tree to single value."""
    result = init
    for _, v in tree_flatten(tree):
        result = fn(result, v)
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SFT training for reasoning models (MLX/Apple Silicon)")
    parser.add_argument("--model_name", default="Qwen/Qwen2-0.5B-Instruct-MLX")
    parser.add_argument("--dataset_name", default="openai/gsm8k",
                        help="Dataset name (auto-selects preparation module)")
    parser.add_argument("--output_dir", default="./output_mlx")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    model, _ = load(args.model_name)
    # Loading like this Solves TypeError: 'TokenizerWrapper' object is not callable
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    train_dataset, val_dataset = prepare_datasets(args, tokenizer)
    
    train_loader = MLXDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = MLXDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Show sample
    sample = tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False)
    logging.info(f"\nSample ({len(train_dataset[0]['input_ids'])} tokens):\n{sample[:500]}...\n")
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = min(steps_per_epoch * args.num_epochs, args.max_steps) \
                    if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    
    optimizer, lr_schedule = create_optimizer_and_scheduler(model, args, total_steps)
    
    model = train(model, tokenizer, model_config, train_loader, val_loader, optimizer, lr_schedule, args)
    
    # Save final model
    final_dir = Path(args.output_dir) / "final_model"
    save_model(model, tokenizer, model_config, final_dir)
    
    logging.info(f"\n{'='*80}\nFinal model saved to {final_dir}\n{'='*80}")


if __name__ == "__main__":
    main()
