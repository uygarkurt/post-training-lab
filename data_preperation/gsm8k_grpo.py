"""
Data loading for GRPO training on openai/gsm8k (socratic split).

Each sample is a dict with prompt token ids and the ground-truth numeric answer
extracted from the dataset's #### final-answer format.

Public API
----------
build_grpo_samples(tokenizer, args) -> list[dict]
    Returns shuffled train samples ready for the GRPO rollout loop.
"""

import random
import re

from datasets import load_dataset as hf_load_dataset


DATASET_NAME   = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT  = "train"

_ANSWER_RE = re.compile(r"####\s*([\d,.]+)")


def extract_gsm8k_answer(text: str) -> str | None:
    """Parse the final numeric answer after ####, stripping commas."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def build_grpo_samples(tokenizer, args) -> list[dict]:
    """
    Tokenise GSM8K questions and return GRPO-ready train samples.

    Each sample is a dict:
        {"prompt_ids": List[int], "ground_truth": str, "question": str}

    Parameters
    ----------
    tokenizer : HF tokenizer with apply_chat_template and encode.
    args      : parsed argparse.Namespace from grpo_train_mlx.py.
    """
    ds = hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

    samples = []
    skipped = 0

    for row in ds:
        question = row["question"]
        answer   = row["answer"]

        ground_truth = extract_gsm8k_answer(answer)
        if ground_truth is None:
            skipped += 1
            continue

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text)

        if len(prompt_ids) >= args.max_prompt_len:
            skipped += 1
            continue

        samples.append({
            "prompt_ids": prompt_ids,
            "ground_truth": ground_truth,
            "question": question,
        })

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_val = max(1, int(len(samples) * args.val_split))
    train = samples[n_val:]

    print(f"  {len(samples)} samples loaded, {skipped} skipped  "
          f"→  {len(train)} train / {n_val} val  "
          f"({args.val_split * 100:.0f}% val split).")

    return train
