"""GSM8K validation for native MLX models."""

from tqdm import tqdm

from data_preparation import gsm8k
from mlx_backend.generate_text import generate_completions


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
    val_loader = gsm8k.build_grpo_dataloader(
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
