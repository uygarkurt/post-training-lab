"""
Shared preparation for openai/gsm8k (socratic split).

This module owns tokenization, dataset splitting, PyTorch Dataset/DataLoader
construction, padding, and batching. Backend entrypoints are responsible for
converting returned batches to their native tensors and running model code.

Public API
----------
build_sft_samples(...)
    Return train/validation lists of (input_ids, loss_mask) pairs.

build_sft_dataloaders(...)
    Return PyTorch DataLoaders that yield padded input and mask tensors.

build_grpo_samples(...)
    Return train/validation lists of prompt dictionaries.

build_debug_overfit_samples(...)
    Return matching tiny train/validation lists for GRPO smoke tests.

answer_rewards(...)
    Return GSM8K answer-correctness rewards for decoded model completions.
"""

import random
import re

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset


DATASET_NAME = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT = "train"

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:)\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_number(raw: str) -> str:
    value = raw.strip().replace(",", "")
    if value.endswith("."):
        value = value[:-1]
    return value


def extract_gsm8k_answer(text: str) -> str | None:
    """Parse the final numeric answer after the GSM8K #### marker."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return _normalize_number(match.group(1))


def extract_final_answer(text: str) -> str | None:
    """Extract the most likely final numeric answer from model output."""
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


def answers_match(prediction: str | None, ground_truth: str) -> bool:
    """Return whether two answers match, using numeric comparison when possible."""
    if prediction is None:
        return False
    try:
        return abs(float(prediction) - float(ground_truth)) < 1e-5
    except ValueError:
        return prediction == ground_truth


def answer_rewards(rollouts_text: list[str], ground_truth: str) -> list[float]:
    """Return a binary correctness reward for each generated completion."""
    rewards = []

    for text in rollouts_text:
        predicted_answer = extract_final_answer(text)
        is_correct = answers_match(predicted_answer, ground_truth)
        rewards.append(1.0 if is_correct else 0.0)

    return rewards


def _load_rows():
    return hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)


def build_sft_samples(
    tokenizer,
    *,
    max_seq_len: int,
    val_split: float,
    seed: int,
) -> tuple[
    list[tuple[list[int], list[int]]],
    list[tuple[list[int], list[int]]],
]:
    """Tokenize GSM8K and return backend-neutral SFT train/validation samples."""
    samples = []
    skipped = 0

    for row in _load_rows():
        question = row["question"]
        answer = row["answer"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)

        if len(full_ids) < 4 or len(prompt_ids) >= len(full_ids):
            skipped += 1
            continue

        full_ids = full_ids[:max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

        if sum(loss_mask) == 0:
            skipped += 1
            continue

        samples.append((full_ids, loss_mask))

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_val = max(1, int(len(samples) * val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(
        f"  {len(samples)} samples loaded, {skipped} skipped  "
        f"→  {len(train_samples)} train / {len(val_samples)} val  "
        f"({val_split * 100:.0f}% val split)."
    )
    return train_samples, val_samples


class GSM8KDataset(Dataset):
    """Hold tokenized (input_ids, loss_mask) samples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _make_sft_collate_fn(pad_id):
    """Build a collator that pads samples into PyTorch tensors."""
    def collate_fn(batch):
        batch_ids = [item[0] for item in batch]
        batch_masks = [item[1] for item in batch]
        max_len = max(len(input_ids) for input_ids in batch_ids)

        padded_ids = []
        padded_masks = []
        for input_ids, loss_mask in zip(batch_ids, batch_masks):
            pad = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_id] * pad)
            padded_masks.append(loss_mask + [0] * pad)

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(padded_masks, dtype=torch.float32),
        )

    return collate_fn


def build_sft_dataloaders(
    tokenizer,
    *,
    max_seq_len: int,
    val_split: float,
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic PyTorch DataLoaders for SFT."""
    train_samples, val_samples = build_sft_samples(
        tokenizer,
        max_seq_len=max_seq_len,
        val_split=val_split,
        seed=seed,
    )
    collate = _make_sft_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        GSM8KDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        GSM8KDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def _load_grpo_samples(
    tokenizer,
    *,
    max_prompt_len: int,
    seed: int,
) -> tuple[list[dict], int]:
    samples = []
    skipped = 0

    for row in _load_rows():
        question = row["question"]
        answer = row["answer"]

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

        if len(prompt_ids) >= max_prompt_len:
            skipped += 1
            continue

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "ground_truth": ground_truth,
                "question": question,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples, skipped


def build_grpo_samples(
    tokenizer,
    *,
    max_prompt_len: int,
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Tokenize GSM8K and return backend-neutral GRPO train/validation samples."""
    samples, skipped = _load_grpo_samples(
        tokenizer,
        max_prompt_len=max_prompt_len,
        seed=seed,
    )

    n_val = max(1, int(len(samples) * val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(
        f"  {len(samples)} samples loaded, {skipped} skipped  "
        f"→  {len(train_samples)} train / {len(val_samples)} val  "
        f"({val_split * 100:.0f}% val split)."
    )
    return train_samples, val_samples


def build_debug_overfit_samples(
    tokenizer,
    *,
    max_prompt_len: int,
    seed: int,
    debug_samples: int,
) -> tuple[list[dict], list[dict]]:
    """Return matching tiny train/validation lists for a GRPO overfit test."""
    samples, skipped = _load_grpo_samples(
        tokenizer,
        max_prompt_len=max_prompt_len,
        seed=seed,
    )
    subset = samples[: min(debug_samples, len(samples))]

    print(
        f"  {len(samples)} samples loaded, {skipped} skipped  "
        f"→  debug overfit: {len(subset)} GSM8K samples "
        "(same set for train and val)."
    )
    return subset, list(subset)
