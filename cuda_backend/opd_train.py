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
    """Parse CUDA on-policy distillation options."""
    parser = argparse.ArgumentParser(
        description="On-policy distillation on CUDA (NVIDIA GPU).",
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
        choices=("xlam-tool-calling-mixed", "gsm8k"),
        default="gsm8k",
        help="Prompt dataset for on-policy distillation",
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

    parser.add_argument(
        "--teacher-model", type=str, required=True,
        help="Frozen teacher model with the same token IDs and output vocabulary as --model",
    )

    parser.add_argument("--num-rollouts", type=int, default=4, help="Number of student rollouts per prompt")
    parser.add_argument("--max-new-tok", type=int, default=256, help="Max tokens to generate per rollout")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate")
    parser.add_argument("--num-iters", type=int, default=500, help="Number of outer training iterations")

    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale = alpha / rank)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization, data shuffle, and rollout sampling")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out from the selected training dataset")
    parser.add_argument("--max-prompt-len", type=int, default=640, help="Skip selected-dataset prompts longer than this")
    parser.add_argument("--log-every", type=int, default=10, help="Write aggregated training metrics to TensorBoard every N steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Validate every N steps after the initial validation (-1 to disable)")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Prompts generated together during validation")

    parser.add_argument("--save-every", type=int, default=100, help="Save a model checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/cuda/opd", help="Base path for timestamped checkpoint directories")

    args = parser.parse_args()
    if args.num_rollouts < 1:
        parser.error("--num-rollouts must be at least 1")
    if args.num_iters < 1:
        parser.error("--num-iters must be at least 1")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
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


def load_policy_and_teacher(args):
    """Load the trainable student and a compatible frozen teacher on CUDA."""
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

    # Keep dropout disabled; evaluation mode still allows student gradients.
    policy.eval()

    print(f"Loading teacher {args.teacher_model} ...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher must use the same token-to-ID mapping.")

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype="bfloat16",
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
        f"Training mode: {args.train_mode} | "
        f"{trainable_parameters:,} / {total_parameters:,} parameters trainable"
    )

    return policy, teacher, tokenizer


def load_opd_dataset(tokenizer, args):
    """Load the selected prompt dataset with a deterministic runtime split."""
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
def generate_rollouts(policy, prompt_tensor, num_rollouts, max_new_tok, tokenizer):
    # All allowing mask. Removes EOS and PAD token ambiguity.
    prompt_attention_mask = torch.ones_like(prompt_tensor)
    generated_ids = policy.generate(
        input_ids=prompt_tensor,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tok,
        num_return_sequences=num_rollouts,
        **ROLLOUT_SAMPLING,
        use_cache=True
    ) # [G, P + L]
    
    prompt_length = prompt_tensor.shape[1]  # P
    completion_ids = generated_ids[:, prompt_length:]  # [G, L]

    # Works for Qwen. It produces EOS after generation ends. Fills the rest with PAD.
    is_eos_or_pad = (completion_ids == tokenizer.eos_token_id) | (completion_ids == tokenizer.pad_token_id)

    # We have to include the first EOS token
    completion_mask = (is_eos_or_pad.cumsum(dim=-1) <= 1).long()  # [G, L]
    truncated_rollouts = ~is_eos_or_pad.any(dim=-1)  # [G]

    return completion_ids, completion_mask, truncated_rollouts

def validate_policy(policy, val_dataset, tokenizer, args, writer, step, progress=None):
    """Run and log the validation metrics for the selected prompt dataset."""
    if args.dataset == "xlam-tool-calling-mixed":
        metrics = xlam_function_calling_eval.validate(
            policy,
            val_dataset,
            tokenizer,
            args.max_new_tok,
            args.eval_batch_size,
        )
        writer.add_scalar("val/exact tool-call accuracy", metrics["accuracy"], step)
        writer.add_scalar("val/function name accuracy", metrics["name_accuracy"], step)
        call_accuracy = metrics.get("call_accuracy", float("nan"))
        no_call_accuracy = metrics.get("no_call_accuracy", float("nan"))
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
    """Run on-policy distillation; N rollouts, P prompt tokens, L completion tokens, V vocabulary."""
    args = parse_args()
    saved_args = vars(args).copy()
    saved_args["rollout_sampling"] = ROLLOUT_SAMPLING.copy()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = f"opd_{timestamp}"
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
    writer = None
    progress = None
    try:
        os.dup2(tee_process.stdin.fileno(), 1)
        os.dup2(tee_process.stdin.fileno(), 2)
        print(f"Run logs directory: {run_logs_dir}")
        print(f"Checkpoint directory: {args.checkpoint_dir}")
        print(f"Rollout sampling: {ROLLOUT_SAMPLING}")

        set_random_seed(args.seed)
        policy, teacher, tokenizer = load_policy_and_teacher(args)
        trainable_parameters = [
            parameter
            for parameter in policy.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
        train_samples, val_dataset = load_opd_dataset(tokenizer, args)
        writer = SummaryWriter(log_dir=run_logs_dir)
        print(f"TensorBoard logs: tensorboard --logdir={run_logs_dir}")

        if args.eval_every != -1:
            validate_policy(
                policy, val_dataset, tokenizer, args, writer, step=0,
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
        gradient_norm_sum_since_log = 0.0
        truncated_rollout_fraction_sum_since_log = 0.0
        completion_tokens_sum_since_log = 0
        elapsed_time_since_log = 0.0

        for step in range(total_steps):
            epoch_number, batch_offset = divmod(step, batches_per_epoch)
            batch_index = batch_offset + 1
            sample = train_samples[batch_offset]
            step_start_time = time.time()
            prompt_tensor = torch.tensor(
                sample["prompt_ids"], dtype=torch.long, device="cuda",
            ).unsqueeze(0)  # [1, P]

            rollouts, rollout_masks, truncated_rollouts = generate_rollouts(
                policy,
                prompt_tensor,
                args.num_rollouts,
                args.max_new_tok,
                tokenizer,
            )  # [N, L], [N, L], [N]

            prompt_ids_extended = torch.repeat_interleave(prompt_tensor, repeats=rollouts.size(0), dim=0) # [N, P]
            complete_ids = torch.cat([prompt_ids_extended, rollouts], dim=-1) # [N, P + L]

            student_logits = policy(complete_ids, use_cache=False).logits # [N, P + L, V]

            # Take the last prompt token, the rest is regular logits shift
            student_logits_shift = student_logits[:, prompt_tensor.size(1) - 1:-1, :] # [N, L, V]

            with torch.no_grad():
                teacher_logits = teacher(complete_ids, use_cache=False).logits # [N, P + L, V]
                teacher_logits_shift = teacher_logits[:, prompt_tensor.size(1) - 1:-1, :] # [N, L, V]

            student_logprobs = torch.log_softmax(
                student_logits_shift, dim=-1, dtype=torch.float32,
            ) # [N, L, V]
            teacher_logprobs = torch.log_softmax(
                teacher_logits_shift, dim=-1, dtype=torch.float32,
            ) # [N, L, V]

            # Summing over V gives KL div values for every position
            kl = F.kl_div(
                teacher_logprobs,
                student_logprobs,
                log_target=True,
                reduction="none",
            ).sum(dim=-1) # [N, L]

            opd_loss = (kl * rollout_masks).sum() / rollout_masks.sum()

            optimizer.zero_grad()
            opd_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )
            
            optimizer.step()

            completed_step = step + 1
            completed_steps = completed_step
            mean_loss = opd_loss.detach().float().item()
            mean_gradient_norm = gradient_norm.detach().float().item()
            truncated_rollout_fraction = truncated_rollouts.float().mean().item()
            completion_tokens = rollout_masks.sum().item()
            step_elapsed_time = time.time() - step_start_time

            log_step_count += 1
            loss_sum_since_log += mean_loss
            gradient_norm_sum_since_log += mean_gradient_norm
            truncated_rollout_fraction_sum_since_log += truncated_rollout_fraction
            completion_tokens_sum_since_log += completion_tokens
            elapsed_time_since_log += step_elapsed_time

            if completed_step % args.log_every == 0 or completed_step == total_steps:
                writer.add_scalar(
                    "train/loss", loss_sum_since_log / log_step_count, completed_step,
                )
                writer.add_scalar(
                    "train/grad_norm",
                    gradient_norm_sum_since_log / log_step_count,
                    completed_step,
                )
                writer.add_scalar(
                    "train/truncated_rollout_fraction",
                    truncated_rollout_fraction_sum_since_log / log_step_count,
                    completed_step,
                )
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], completed_step,
                )
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
                gradient_norm_sum_since_log = 0.0
                truncated_rollout_fraction_sum_since_log = 0.0
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
                    policy, val_dataset, tokenizer, args, writer, completed_step,
                    progress=progress,
                )

            if args.save_every > 0 and completed_step % args.save_every == 0:
                writer.flush()
                save_checkpoint(
                    policy, tokenizer, args.checkpoint_dir, completed_step,
                    training_start_time, optimizer.param_groups[0]["lr"],
                )

        if (
            args.save_every > 0
            and completed_steps > 0
            and completed_steps % args.save_every != 0
        ):
            writer.flush()
            save_checkpoint(
                policy, tokenizer, args.checkpoint_dir, completed_steps,
                training_start_time, optimizer.param_groups[0]["lr"],
            )
    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
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
