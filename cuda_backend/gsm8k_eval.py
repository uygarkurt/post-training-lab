"""GSM8K evaluation for CUDA models."""

import argparse

import torch
from tqdm import tqdm

from cuda_backend.generate_text import load_model
from data_preparation import gsm8k


def validate(
    policy,
    val_dataset,
    tokenizer,
    max_new_tokens,
    batch_size,
    description="val",
):
    """Calculate accuracy by greedily generating over a batched dataset."""
    correct_answers = 0
    total_answers = 0
    device = next(policy.parameters()).device
    val_loader = gsm8k.build_grpo_dataloader(
        val_dataset,
        pad_id=tokenizer.pad_token_id,
        batch_size=batch_size,
    )

    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc=f"  {description}",
            leave=False,
            unit="batch",
        ):
            prompt_tensor = batch["prompt_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)

            generated_ids = policy.generate(
                input_ids=prompt_tensor,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                use_cache=True,
            )

            prompt_length = prompt_tensor.shape[1]
            completion_ids = generated_ids[:, prompt_length:]
            completion_texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )

            correct_answers += sum(
                gsm8k.is_answer_correct(text, ground_truth)
                for text, ground_truth in zip(
                    completion_texts,
                    batch["ground_truth"],
                )
            )
            total_answers += len(batch["ground_truth"])

    if total_answers == 0:
        return float("nan")
    return correct_answers / total_answers


def main():
    """Load a CUDA checkpoint and evaluate it on the GSM8K test split."""
    parser = argparse.ArgumentParser(
        description="Evaluate a CUDA model on the GSM8K test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Path to a Hugging Face model or checkpoint directory",
    )
    parser.add_argument(
        "--load-adapter",
        action="store_true",
        help="Load a PEFT adapter from --model_path",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=512,
        help="Skip GSM8K prompts longer than this",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per answer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts generated together during evaluation",
    )
    args = parser.parse_args()

    policy, tokenizer = load_model(args.model_path, args.load_adapter)
    policy.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading GSM8K test split ...")
    test_dataset = gsm8k.GSM8KGRPODataset(
        tokenizer,
        max_prompt_len=args.max_prompt_len,
        split="test",
    )
    print(
        f"  {len(test_dataset)} samples loaded, "
        f"{test_dataset.skipped} skipped."
    )

    test_accuracy = validate(
        policy,
        test_dataset,
        tokenizer,
        args.max_new_tokens,
        args.batch_size,
        description="test",
    )
    print(f"GSM8K test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
