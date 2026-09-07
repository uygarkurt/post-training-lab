import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_preparation import gsm8k, numinamath


def parse_args():
    """Parse the small set of options needed by the SFT scaffold."""
    parser = argparse.ArgumentParser(
        description="SFT for CUDA (NVIDIA GPU).",
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
        choices=("numinamath-algebra", "gsm8k"),
        default="numinamath-algebra",
        help="SFT dataset to train on",
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

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization and data shuffling")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from the training dataset")
    parser.add_argument("--log-every", type=int, default=10, help="Write training metrics to TensorBoard every N steps (loss averaged over response tokens)")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")

    parser.add_argument("--tensorboard-dir", type=str, default="./runs/cuda/sft", help="Base path for timestamped TensorBoard run directories")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/cuda/sft", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.debug_samples < 1:
        parser.error("--debug-samples must be at least 1")
    if args.num_iters < 1:
        parser.error("--num-iters must be at least 1")
    if args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    return args


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch for deterministic CUDA execution."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(args):
    """Load a dense model or trainable adapter using the CUDA training conventions."""
    print(f"Loading model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=continue_adapter_training,
        )

        if args.train_mode == "full":
            print("Merging adapter into the model for full training ...")
            model = model.merge_and_unload()
            model.requires_grad_(True)

    elif args.train_mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
        )
        print("Applying new LoRA adapters to all linear layers ...")
        model = get_peft_model(model, lora_config)

    else:
        model.requires_grad_(True)

    model.config.use_cache = False

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training mode: {args.train_mode} | "
        f"{trainable_parameters:,} / {total_parameters:,} parameters trainable"
    )
    return model, tokenizer


def load_sft_datasets(tokenizer, args):
    """Load the selected SFT datasets and return their matching collator builder."""
    if args.dataset == "numinamath-algebra":
        print("Loading NuminaMath Algebra dataset ...")
        dataset_class = numinamath.NuminaMathSFTDataset
        build_dataloader = numinamath.build_sft_dataloader
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


@torch.inference_mode()
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
        input_ids = input_ids.to("cuda")  # [B, L]
        loss_mask = loss_mask.to("cuda")  # [B, L]

        logits = model(input_ids).logits  # [B, L, V]

        logits_shifted = logits[:, :-1, :]  # [B, L-1, V]
        targets = input_ids[:, 1:]  # [B, L-1]
        loss_mask_shifted = loss_mask[:, 1:]  # [B, L-1]

        ce = F.cross_entropy(
            logits_shifted.transpose(1, 2),  # [B, V, L-1]
            targets,
            reduction="none",
        )  # [B, L-1]

        masked_ce = ce * loss_mask_shifted
        response_token_count = int(loss_mask_shifted.sum().item())

        total_loss += masked_ce.float().sum().item()
        total_response_token_count += response_token_count

    if was_training:
        model.train()

    if total_response_token_count == 0:
        return float("nan")
    return total_loss / total_response_token_count


def save_checkpoint(model, tokenizer, checkpoint_dir, step, training_start_time):
    """Save model, tokenizer, and elapsed hours, including eval and saves."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:06d}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    metadata = {
        "step": step,
        "elapsed_hours": (time.monotonic() - training_start_time) / 3600,
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

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs: tensorboard --logdir={args.tensorboard_dir}")

    if args.eval_every != -1:
        validation_loss = calculate_validation_loss(model, val_loader)
        writer.add_scalar("val/loss", validation_loss, 0)
        print(f"  [val] step {0:5d} | loss {validation_loss:.4f}")

    model.train()
    batches_per_epoch = len(train_loader)
    progress = tqdm(
        total=args.num_iters,
        desc=f"epoch=1 batch=0/{batches_per_epoch} loss=----",
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
            input_ids = input_ids.to("cuda")  # [B, L]
            loss_mask = loss_mask.to("cuda")  # [B, L]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids).logits  # [B, L, V]

            logits_shifted = logits[:, :-1, :]  # [B, L-1, V]
            targets = input_ids[:, 1:]  # [B, L-1]
            loss_mask_shifted = loss_mask[:, 1:]  # [B, L-1]

            ce = F.cross_entropy(
                logits_shifted.transpose(1, 2),
                targets,
                reduction="none",
            )  # [B, L-1]

            masked_ce = ce * loss_mask_shifted  # [B, L-1]
            response_token_count = loss_mask_shifted.sum()
            loss = masked_ce.sum() / response_token_count

            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            loss_value = loss.detach().float().item()
            gradient_norm_value = gradient_norm.detach().float().item()
            response_token_count = int(response_token_count.detach().item())
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
                writer.add_scalar("train/learning_rate", args.lr, step)
                writer.add_scalar("train/response_tokens", response_token_count, step)
                writer.add_scalar("train/tokens_per_sec", tokens_per_second, step)
                writer.add_scalar("train/epoch", dataset_passes, step)
                writer.add_scalar("train/examples_seen", examples_seen, step)
                loss_sum_since_log = 0.0
                response_tokens_since_log = 0

            progress.set_description(
                f"epoch={epoch_number} batch={batch_index}/{batches_per_epoch} "
                f"loss={loss_value:.4f}",
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
