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

GSM8KGRPODataset.build_train_val_datasets(...)
    Return train/validation PyTorch Datasets of prompt dictionaries.

GSM8KGRPODataset.build_debug_overfit_datasets(...)
    Return matching tiny train/validation Datasets for GRPO smoke tests.

build_grpo_dataloader(...)
    Return batches of left-padded GRPO prompts and attention masks.

answer_rewards(...)
    Return GSM8K answer-correctness rewards for decoded model completions.
"""

import random
import re

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split

DATASET_NAME = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT = "train"

_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:)\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _load_rows(split=DATASET_SPLIT):
    return hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=split)


# SFT preparation


class GSM8KSFTDataset(Dataset):
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


def build_sft_samples(
    tokenizer,
    max_seq_len,
    val_split,
    seed,
):
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


def build_sft_dataloaders(
    tokenizer,
    max_seq_len,
    val_split,
    seed,
    batch_size,
):
    """Build deterministic PyTorch DataLoaders for SFT."""
    train_samples, val_samples = build_sft_samples(
        tokenizer,
        max_seq_len=max_seq_len,
        val_split=val_split,
        seed=seed,
    )
    collate = _make_sft_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        GSM8KSFTDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        GSM8KSFTDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


# GRPO preparation and rewards


def _normalize_number(raw):
    """Remove whitespace, thousands separators, and a trailing decimal point."""
    value = raw.strip().replace(",", "")
    return value.removesuffix(".")


def extract_gsm8k_answer(text):
    """Parse the final numeric answer after the GSM8K #### marker."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    return _normalize_number(match.group(1))


def extract_final_answer(text):
    """Extract the most likely final numeric answer from a model completion."""
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


def answers_match(prediction, ground_truth):
    """Compare two answers numerically when possible."""
    if prediction is None:
        return False

    try:
        return abs(float(prediction) - float(ground_truth)) < 1e-5
    except ValueError:
        return prediction == ground_truth


def is_answer_correct(completion_text, ground_truth):
    """Return whether a model completion contains the expected final answer."""
    predicted_answer = extract_final_answer(completion_text)
    return answers_match(predicted_answer, ground_truth)


def answer_rewards(rollouts_text, ground_truth):
    """Return binary rewards for decoded model completions."""
    rewards = []

    for text in rollouts_text:
        is_correct = is_answer_correct(text, ground_truth)
        rewards.append(1.0 if is_correct else 0.0)

    return rewards


def _make_grpo_collate_fn(pad_id):
    """Build a collator that left-pads GRPO prompts for generation."""
    def collate_fn(batch):
        max_prompt_length = max(len(sample["prompt_ids"]) for sample in batch)
        prompt_ids = []
        prompt_attention_masks = []

        for sample in batch:
            padding_length = max_prompt_length - len(sample["prompt_ids"])
            prompt_ids.append(
                [pad_id] * padding_length + sample["prompt_ids"]
            )
            prompt_attention_masks.append(
                [0] * padding_length + [1] * len(sample["prompt_ids"])
            )

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(
                prompt_attention_masks,
                dtype=torch.long,
            ),
            "ground_truth": [sample["ground_truth"] for sample in batch],
        }

    return collate_fn


def build_grpo_dataloader(dataset, pad_id, batch_size):
    """Build a deterministic DataLoader of padded GRPO prompts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_grpo_collate_fn(pad_id),
        num_workers=0,
        drop_last=False,
    )


class GSM8KGRPODataset(Dataset):
    """Tokenize and hold GSM8K prompts and answers for GRPO."""

    def __init__(self, tokenizer, max_prompt_len, split=DATASET_SPLIT):
        self.samples = []
        self.skipped = 0

        for row in _load_rows(split):
            question = row["question"]
            answer = row["answer"]

            ground_truth = extract_gsm8k_answer(answer)
            if ground_truth is None:
                self.skipped += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt_text)

            if len(prompt_ids) >= max_prompt_len:
                self.skipped += 1
                continue

            self.samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "ground_truth": ground_truth,
                    "question": question,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @classmethod
    def build_train_val_datasets(
        cls,
        tokenizer,
        max_prompt_len,
        val_split,
        seed,
    ):
        """Build GRPO train and validation datasets."""
        dataset = cls(tokenizer, max_prompt_len=max_prompt_len)

        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        print(
            f"  {len(dataset)} samples loaded, {dataset.skipped} skipped  "
            f"→  {len(train_dataset)} train / {len(val_dataset)} val  "
            f"({val_split * 100:.0f}% val split)."
        )
        return train_dataset, val_dataset

    @classmethod
    def build_debug_overfit_datasets(
        cls,
        tokenizer,
        max_prompt_len,
        seed,
        debug_samples,
    ):
        """Build matching tiny train and validation datasets."""
        dataset = cls(tokenizer, max_prompt_len=max_prompt_len)

        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        debug_indices = shuffled_indices[: min(debug_samples, len(dataset))]

        train_dataset = Subset(dataset, debug_indices)
        val_dataset = Subset(dataset, debug_indices)

        print(
            f"  {len(dataset)} samples loaded, {dataset.skipped} skipped  "
            f"→  debug overfit: {len(train_dataset)} GSM8K samples "
            "(same set for train and val)."
        )
        return train_dataset, val_dataset
