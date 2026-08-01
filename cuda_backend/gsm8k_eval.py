"""GSM8K validation for CUDA models."""

import torch
from tqdm import tqdm

from data_preparation import gsm8k


def validate(policy, val_dataset, tokenizer, max_new_tokens, batch_size):
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
            desc="  val",
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
