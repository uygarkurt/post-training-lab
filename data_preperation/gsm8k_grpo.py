"""
Data loading for GRPO training on openai/gsm8k (socratic split).

Each sample is a dict with prompt token ids and the ground-truth numeric answer
extracted from the dataset's #### final-answer format.

Public API
----------
build_grpo_samples(tokenizer, args) -> tuple[list[dict], list[dict]]
    Returns (train_samples, val_samples) ready for the GRPO rollout loop.

build_debug_overfit_samples(tokenizer, args) -> tuple[list[dict], list[dict]]
    Returns a tiny GSM8K subset; train and val lists are identical (overfit smoke test).
"""

import random
import re

from datasets import load_dataset as hf_load_dataset


DATASET_NAME   = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT  = "train"

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:)\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_number(raw: str) -> str:
    s = raw.strip().replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_gsm8k_answer(text: str) -> str | None:
    """Parse the final numeric answer after #### (dataset ground truth)."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return _normalize_number(match.group(1))


def extract_final_answer(text: str) -> str | None:
    """
    Extract a predicted final numeric answer from model output.

    Tries, in order:
      1. GSM8K #### marker
      2. 'answer is X' / 'final answer: X' phrasing
      3. Last number appearing anywhere in the text
    """
    if not text or not text.strip():
        return None

    match = _ANSWER_RE.search(text)
    if match:
        return _normalize_number(match.group(1))

    match = _ANSWER_IS_RE.search(text)
    if match:
        return _normalize_number(match.group(1))

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def answers_match(pred: str | None, ground_truth: str) -> bool:
    """Return True if pred matches ground_truth (numeric compare when possible)."""
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(ground_truth)) < 1e-5
    except ValueError:
        return pred == ground_truth


def _load_gsm8k_samples(tokenizer, args) -> tuple[list[dict], int]:
    """Load and tokenise GSM8K; return (samples, skipped_count)."""
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
    return samples, skipped


def build_grpo_samples(tokenizer, args) -> tuple[list[dict], list[dict]]:
    """
    Tokenise GSM8K questions and return GRPO-ready train/val samples.

    Each sample is a dict:
        {"prompt_ids": List[int], "ground_truth": str, "question": str}

    Parameters
    ----------
    tokenizer : HF tokenizer with apply_chat_template and encode.
    args      : parsed argparse.Namespace from grpo_train_mlx.py.

    Returns
    -------
    (train, val) — two lists of sample dicts.
    """
    samples, skipped = _load_gsm8k_samples(tokenizer, args)

    n_val = max(1, int(len(samples) * args.val_split))
    val   = samples[:n_val]
    train = samples[n_val:]

    print(f"  {len(samples)} samples loaded, {skipped} skipped  "
          f"→  {len(train)} train / {len(val)} val  "
          f"({args.val_split * 100:.0f}% val split).")

    return train, val


def build_debug_overfit_samples(tokenizer, args) -> tuple[list[dict], list[dict]]:
    """
    Return a tiny GSM8K subset for debug overfit runs.

    Train and val use the same samples so reward + validation both exercise
    the real GSM8K answer-matching path on a set small enough to overfit.
    """
    samples, skipped = _load_gsm8k_samples(tokenizer, args)
    n = min(args.debug_samples, len(samples))
    subset = samples[:n]

    print(f"  {len(samples)} samples loaded, {skipped} skipped  "
          f"→  debug overfit: {n} GSM8K samples (same set for train and val).")

    return subset, list(subset)
