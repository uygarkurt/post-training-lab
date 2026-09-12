"""xLAM function-calling evaluation for CUDA models."""

import argparse
import json
from contextlib import nullcontext

import torch
from tqdm import tqdm

from cuda_backend.generate_text import load_model
from data_preparation import xlam_function_calling


def validate(
    policy,
    val_dataset,
    tokenizer,
    max_new_tokens,
    batch_size,
    description="val",
    output_file=None,
):
    """Measure exact tool-call and function-name accuracy with greedy decoding."""
    correct_calls = 0
    correct_names = 0
    parsed_answers = 0
    total_answers = 0
    truncated_answers = 0
    device = next(policy.parameters()).device
    eos_token_ids = policy.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    eos_token_ids = torch.as_tensor(
        eos_token_ids if eos_token_ids is not None else [], device=device,
    )
    val_loader = xlam_function_calling.build_eval_dataloader(
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
                predicted_calls, names_correct, calls_correct = (
                    xlam_function_calling.grade_tool_calls(
                        text,
                        batch["answers"][index],
                    )
                )
                truncated = truncated_completions[index]
                parsed_answers += bool(predicted_calls)
                correct_names += names_correct
                correct_calls += calls_correct
                truncated_answers += truncated
                if output_file is not None:
                    output_file.write(json.dumps({
                        "id": batch["id"][index],
                        "query": batch["query"][index],
                        "tools": batch["tools"][index],
                        "answers": batch["answers"][index],
                        "completion": text,
                        "predicted_calls": predicted_calls,
                        "names_correct": names_correct,
                        "correct": calls_correct,
                        "truncated": truncated,
                    }, ensure_ascii=False) + "\n")
            if output_file is not None:
                output_file.flush()
            total_answers += len(batch["answers"])

    if total_answers == 0:
        return {"accuracy": float("nan"), "name_accuracy": float("nan")}
    print(
        f"  {correct_calls}/{total_answers} exact call sets; "
        f"{correct_names}/{total_answers} function-name sets; "
        f"{parsed_answers}/{total_answers} completions parsed; "
        f"{truncated_answers}/{total_answers} reached the token limit without EOS."
    )
    return {
        "accuracy": correct_calls / total_answers,
        "name_accuracy": correct_names / total_answers,
    }


def main():
    """Load a CUDA checkpoint and evaluate it on the local xLAM test split."""
    parser = argparse.ArgumentParser(
        description="Evaluate a CUDA model on the xLAM function-calling test split.",
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
        default=str(xlam_function_calling.TEST_DATASET_PATH),
        help="Path to a prepared local xLAM test JSONL file",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=3072,
        help="Skip tool-aware xLAM prompts longer than this",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per tool-call response",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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
        help="Save prompts, calls, completions, scores, and truncation flags",
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

    print("Loading xLAM function-calling test split ...")
    test_dataset = xlam_function_calling.XLAMFunctionCallingEvalDataset(
        tokenizer,
        max_prompt_len=args.max_prompt_len,
        num_samples=args.num_samples,
        dataset_path=args.dataset_path,
    )
    print(
        f"  {len(test_dataset)} samples loaded, "
        f"{test_dataset.skipped} skipped."
    )
    if test_dataset.skipped_invalid:
        print(f"  {test_dataset.skipped_invalid} invalid rows were discarded.")
    if test_dataset.skipped_overlong:
        print(
            f"  {test_dataset.skipped_overlong} prompts exceeded "
            f"--max-prompt-len {args.max_prompt_len} and were discarded."
        )
    if not test_dataset:
        raise ValueError("No test samples remain; check the data and prompt limit")

    with (
        open(args.output_jsonl, "w", encoding="utf-8")
        if args.output_jsonl else nullcontext()
    ) as output_file:
        metrics = validate(
            policy,
            test_dataset,
            tokenizer,
            args.max_new_tokens,
            args.batch_size,
            description="test",
            output_file=output_file,
        )
    print(f"xLAM exact tool-call accuracy: {metrics['accuracy']:.4f}")
    print(f"xLAM function-name accuracy: {metrics['name_accuracy']:.4f}")
    if args.output_jsonl:
        print(f"Evaluation completions saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
