"""
Shared SFT preparation for the Algebra subset of NuminaMath 1.5.

This module owns dataset filtering, tokenization, splitting, PyTorch
Dataset/DataLoader construction, padding, and batching. Backend entrypoints
remain responsible for model and device operations.

Public API
----------
NuminaMathSFTDataset.build_train_val_datasets(...)
    Return train/validation PyTorch Datasets of Algebra SFT examples.

NuminaMathSFTDataset.build_debug_overfit_datasets(...)
    Return matching tiny train/validation Datasets for SFT smoke tests.

build_sft_dataloader(...)
    Return batches of right-padded SFT examples and loss masks.
"""

from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/numinamath-1.5-rl-verifiable/train.jsonl"
)
DATASET_SPLIT = "train"
PROBLEM_TYPE = "Algebra"


def _make_sft_collate_fn(tokenizer):
    """Build a collator that pads samples into PyTorch tensors."""
    def collate_fn(batch):
        """Pad one list of tokenized SFT samples."""
        padded_batch = tokenizer.pad(
            {
                "input_ids": [sample["input_ids"] for sample in batch],
                "attention_mask": [sample["loss_mask"] for sample in batch],
            },
            padding=True,
            padding_side="right",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return (
            padded_batch["input_ids"],
            padded_batch["attention_mask"].to(torch.float32),
        )

    return collate_fn


def build_sft_dataloader(
    dataset,
    tokenizer,
    batch_size,
    shuffle=False,
    seed=0,
):
    """Build a deterministic, optionally shuffled SFT DataLoader."""
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_make_sft_collate_fn(tokenizer),
        num_workers=0,
        drop_last=False,
    )


class NuminaMathSFTDataset(Dataset):
    """Tokenize and hold complete NuminaMath Algebra examples for SFT."""

    def __init__(self, tokenizer, max_seq_len, split=DATASET_SPLIT):
        self.samples = []
        self.total_rows = 0
        self.algebra_rows = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0

        for row in hf_load_dataset("json", data_files=str(DATASET_PATH), split=split):
            self.total_rows += 1
            if row["problem_type"] != PROBLEM_TYPE:
                continue

            self.algebra_rows += 1
            problem = row["problem"]
            solution = row["solution"]

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer.encode(prompt_text)
            full_ids = tokenizer.encode(full_text)

            if len(full_ids) > max_seq_len:
                self.skipped_overlong += 1
                continue

            if len(full_ids) < 4 or len(prompt_ids) >= len(full_ids):
                self.skipped_invalid += 1
                continue

            loss_mask = [0] * len(prompt_ids) + [1] * (
                len(full_ids) - len(prompt_ids)
            )
            self.samples.append(
                {
                    "input_ids": full_ids,
                    "loss_mask": loss_mask,
                }
            )

    def __len__(self):
        """Return the number of retained Algebra samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized Algebra sample."""
        return self.samples[index]

    def print_summary(self, destination):
        """Print filtering statistics followed by a split description."""
        print(
            f"  {self.total_rows} rows loaded, {self.algebra_rows} Algebra  "
            f"→  {len(self)} retained / {self.skipped_overlong} overlong / "
            f"{self.skipped_invalid} invalid  →  {destination}."
        )

    @classmethod
    def build_train_val_datasets(
        cls,
        tokenizer,
        max_seq_len,
        val_split,
        seed,
    ):
        """Build deterministic NuminaMath Algebra train and validation datasets."""
        dataset = cls(tokenizer, max_seq_len=max_seq_len)

        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        dataset.print_summary(
            f"{len(train_dataset)} train / {len(val_dataset)} val "
            f"({val_split * 100:.0f}% val split)"
        )
        return train_dataset, val_dataset

    @classmethod
    def build_debug_overfit_datasets(
        cls,
        tokenizer,
        max_seq_len,
        seed,
        debug_samples,
    ):
        """Build matching tiny NuminaMath Algebra train and validation datasets."""
        dataset = cls(tokenizer, max_seq_len=max_seq_len)

        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        debug_indices = shuffled_indices[: min(debug_samples, len(dataset))]

        train_dataset = Subset(dataset, debug_indices)
        val_dataset = Subset(dataset, debug_indices)

        dataset.print_summary(
            f"debug overfit: {len(train_dataset)} NuminaMath Algebra samples "
            "(same set for train and val)"
        )
        return train_dataset, val_dataset
