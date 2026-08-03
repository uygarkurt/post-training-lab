"""GSM8K validation for MLX models."""

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

from data_preparation import gsm8k


def validate(policy, val_samples, tokenizer, max_new_tokens):
    """Calculate accuracy over validation samples using greedy generation."""
    if len(val_samples) == 0:
        return float("nan")

    correct_answers = 0
    sampler = make_sampler(temp=0.0)

    for sample in tqdm(val_samples, desc="  val", leave=False, unit="sample"):
        prompt_text = tokenizer.decode(sample["prompt_ids"])
        completion_text = generate(
            policy,
            tokenizer,
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            sampler=sampler,
        )

        if gsm8k.is_answer_correct(completion_text, sample["ground_truth"]):
            correct_answers += 1

    return correct_answers / len(val_samples)
