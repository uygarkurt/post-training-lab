"""MetaMathQA evaluation for CUDA models."""

import argparse
import json
from contextlib import nullcontext

import torch
from tqdm import tqdm

from cuda_backend.generate_text import load_model
from data_preparation import metamathqa


def validate(
    policy,
    val_dataset,
    tokenizer,
    max_new_tokens,
    batch_size,
    description="val",
    output_file=None,
):
    """Calculate accuracy by greedily generating over a batched dataset."""
    correct_answers = 0
    total_answers = 0
    truncated_answers = 0
    device = next(policy.parameters()).device
    eos_token_ids = policy.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    eos_token_ids = torch.as_tensor(
        eos_token_ids if eos_token_ids is not None else [], device=device,
    )
    val_loader = metamathqa.build_eval_dataloader(
        val_dataset,
        tokenizer=tokenizer,
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
                # Clear sampling defaults inherited from the model's config.
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_length = prompt_tensor.shape[1]
            completion_ids = generated_ids[:, prompt_length:]
            completion_texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )

            truncated_completions = (
                ~torch.isin(completion_ids, eos_token_ids).any(dim=-1)
                & (completion_ids.shape[1] >= max_new_tokens)
            ).tolist()
            for index, text in enumerate(completion_texts):
                ground_truth = batch["ground_truth"][index]
                truncated = truncated_completions[index]
                predicted_answer, correct = metamathqa.grade_answer(text, ground_truth)
                correct_answers += correct
                truncated_answers += truncated
                if output_file is not None:
                    output_file.write(json.dumps({
                        "question": batch["question"][index],
                        "answer": batch["answer"][index],
                        "ground_truth": [str(answer) for answer in ground_truth],
                        "completion": text,
                        "predicted_answer": [str(answer) for answer in predicted_answer],
                        "correct": correct,
                        "truncated": truncated,
                    }, ensure_ascii=False) + "\n")
            if output_file is not None:
                output_file.flush()
            total_answers += len(batch["ground_truth"])

    if total_answers == 0:
        return float("nan")
    print(
        f"  {correct_answers}/{total_answers} answers correct; "
        f"{truncated_answers}/{total_answers} reached the token limit without EOS."
    )
    return correct_answers / total_answers


def main():
    """Load a CUDA checkpoint and evaluate it on the MetaMathQA test split."""
    parser = argparse.ArgumentParser(
        description="Evaluate a CUDA model on the MetaMathQA test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to a Hugging Face model or checkpoint directory",
    )
    parser.add_argument(
        "--load-adapter",
        action="store_true",
        help="Load a PEFT adapter from --model_path",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(metamathqa.TEST_DATASET_PATH),
        help="Path to a prepared local MetaMathQA test JSONL file",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=512,
        help="Skip MetaMathQA prompts longer than this",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per answer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts generated together during evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate the first N usable test samples; omit to evaluate all",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Save questions, references, completions, parsed predictions, correctness, and truncation flags",
    )
    args = parser.parse_args()
    if args.max_prompt_len < 1 or args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("Token limits and batch size must be positive")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")

    policy, tokenizer = load_model(args.model_path, args.load_adapter)
    policy.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading MetaMathQA test split ...")
    test_dataset = metamathqa.MetaMathQAEvalDataset(
        tokenizer,
        max_prompt_len=args.max_prompt_len,
        num_samples=args.num_samples,
        dataset_path=args.dataset_path,
    )
    print(
        f"  {len(test_dataset)} samples loaded, "
        f"{test_dataset.skipped} skipped."
    )

    if test_dataset.empty_queries:
        print(f"  {test_dataset.empty_queries} empty queries were discarded.")
    if test_dataset.skipped_overlong:
        print(
            f"  {test_dataset.skipped_overlong} prompts exceeded "
            f"--max-prompt-len {args.max_prompt_len} and were discarded."
        )
    if test_dataset.unparsed_answers:
        print(
            f"  {test_dataset.unparsed_answers} reference answers could not be "
            "parsed; these were discarded from evaluation."
        )
    if not test_dataset:
        raise ValueError(
            "No test samples remain; check reference answers and --max-prompt-len"
        )

    with (
        open(args.output_jsonl, "w", encoding="utf-8")
        if args.output_jsonl else nullcontext()
    ) as output_file:
        test_accuracy = validate(
            policy,
            test_dataset,
            tokenizer,
            args.max_new_tokens,
            args.batch_size,
            description="test",
            output_file=output_file,
        )
    print(f"MetaMathQA test accuracy: {test_accuracy:.4f}")
    if args.output_jsonl:
        print(f"Evaluation completions saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
