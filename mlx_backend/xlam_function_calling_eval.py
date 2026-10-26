"""xLAM/irrelevance function-calling validation for native MLX models."""

import json

from tqdm import tqdm

from data_preparation import xlam_function_calling
from mlx_backend.generate_text import generate_completions


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
    no_call_answers = 0
    correct_no_call_answers = 0
    no_tool_answers = 0
    correct_no_tool_answers = 0
    irrelevant_tool_answers = 0
    correct_irrelevant_tool_answers = 0
    call_answers = 0
    correct_call_answers = 0
    val_loader = xlam_function_calling.build_eval_dataloader(
        val_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    for batch in tqdm(
        val_loader,
        desc=f"  {description}",
        leave=False,
        unit="batch",
    ):
        completion_ids = generate_completions(
            policy,
            batch["prompt_ids"],
            batch["prompt_attention_mask"],
            tokenizer,
            max_new_tokens,
        )
        completion_texts = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        truncated_completions = [
            not any(token in tokenizer.eos_token_ids for token in ids)
            and len(ids) >= max_new_tokens
            for ids in completion_ids
        ]

        for index, text in enumerate(completion_texts):
            predicted_calls, names_correct, calls_correct = (
                xlam_function_calling.tool_call_reward(
                    text,
                    batch["answers"][index],
                )
            )
            truncated = truncated_completions[index]
            parsed_answers += bool(predicted_calls)
            correct_names += names_correct
            correct_calls += calls_correct
            truncated_answers += truncated
            if batch["answers"][index]:
                call_answers += 1
                correct_call_answers += calls_correct
            else:
                no_call_answers += 1
                correct_no_call_answers += calls_correct
                if batch["tools"][index]:
                    irrelevant_tool_answers += 1
                    correct_irrelevant_tool_answers += calls_correct
                else:
                    no_tool_answers += 1
                    correct_no_tool_answers += calls_correct
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
        f"{parsed_answers}/{total_answers} completions with parsed calls; "
        f"{truncated_answers}/{total_answers} reached the token limit without EOS."
    )
    if no_call_answers:
        print(
            f"  {correct_no_call_answers}/{no_call_answers} correct no-call responses; "
            f"{correct_call_answers}/{call_answers} correct call responses."
        )
    if no_tool_answers:
        print(
            f"  {correct_no_tool_answers}/{no_tool_answers} correct with no tools; "
            f"{correct_irrelevant_tool_answers}/{irrelevant_tool_answers} "
            "correct with irrelevant tools."
        )
    return {
        "accuracy": correct_calls / total_answers,
        "name_accuracy": correct_names / total_answers,
        "no_call_accuracy": (
            correct_no_call_answers / no_call_answers if no_call_answers else float("nan")
        ),
        "no_tool_accuracy": (
            correct_no_tool_answers / no_tool_answers if no_tool_answers else float("nan")
        ),
        "irrelevant_tool_accuracy": (
            correct_irrelevant_tool_answers / irrelevant_tool_answers
            if irrelevant_tool_answers else float("nan")
        ),
        "call_accuracy": correct_call_answers / call_answers if call_answers else float("nan"),
    }

