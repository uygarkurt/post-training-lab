"""
Data loading and batching for openai/gsm8k (socratic split).

Each sample is a chat-formatted (input_ids, loss_mask) pair where:
    loss_mask = 0  → prompt token   (no gradient)
    loss_mask = 1  → answer token   (supervised)

Public API
----------
build_dataloaders(tokenizer, args) -> (train_loader, val_loader)
    Returns two PyTorch DataLoaders whose batches are (mx.array, mx.array)
    tuples of shape [B, T] ready for the MLX training loop.
"""

import random

import mlx.core as mx
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset


DATASET_NAME   = "openai/gsm8k"
DATASET_SUBSET = "socratic"
DATASET_SPLIT  = "train"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GSM8KDataset(Dataset):
    """Holds pre-tokenised (input_ids, loss_mask) pairs as Python lists."""

    def __init__(self, samples):
        self.samples = samples  # list of (List[int], List[int])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (input_ids, loss_mask)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _make_collate_fn(pad_id):
    """
    Returns a collate function that right-pads sequences in a batch to the
    same length and converts them to MLX arrays.
    """
    def collate_fn(batch):
        batch_ids  = [item[0] for item in batch]
        batch_msk  = [item[1] for item in batch]
        max_len    = max(len(ids) for ids in batch_ids)

        padded_ids, padded_masks = [], []
        for ids, mask in zip(batch_ids, batch_msk):
            pad = max_len - len(ids)
            padded_ids.append(ids   + [pad_id] * pad)
            padded_masks.append(mask + [0]     * pad)

        return (
            mx.array(padded_ids,   dtype=mx.int32),
            mx.array(padded_masks, dtype=mx.float32),
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_dataloaders(tokenizer, args):
    """
    Tokenise the dataset, split into train/val, and return two DataLoaders.

    Each batch yielded by the loaders is a tuple:
        (input_ids: mx.array [B, T], loss_mask: mx.array [B, T])

    Parameters
    ----------
    tokenizer : HF tokenizer with apply_chat_template and encode.
    args      : parsed argparse.Namespace (see parse_args in sft_train_mlx.py).
    """
    ds = hf_load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

    samples = []
    skipped = 0

    for row in ds:
        question = row["question"]
        answer   = row["answer"]

        # Render the user prompt so we can find where the answer begins.
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full conversation: user question + assistant CoT answer.
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer.encode(prompt_text)
        full_ids   = tokenizer.encode(full_text)

        # Skip degenerate samples.
        if len(full_ids) < 4 or len(prompt_ids) >= len(full_ids):
            skipped += 1
            continue

        # Truncate to max_seq_len.
        full_ids = full_ids[:args.max_seq_len]

        # Build binary loss mask aligned with full_ids.
        # Everything up to (and including) the prompt gets 0; the rest gets 1.
        prompt_len = min(len(prompt_ids), len(full_ids))
        loss_mask  = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
        loss_mask  = loss_mask[:args.max_seq_len]

        # Skip if the answer was entirely cropped away.
        if sum(loss_mask) == 0:
            skipped += 1
            continue

        samples.append((full_ids, loss_mask))

    # Shuffle deterministically before splitting so the val set is stable.
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_val  = max(1, int(len(samples) * args.val_split))
    val    = samples[:n_val]
    train  = samples[n_val:]

    print(f"  {len(samples)} samples loaded, {skipped} skipped  "
          f"→  {len(train)} train / {len(val)} val  "
          f"({args.val_split * 100:.0f}% val split).")

    collate = _make_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        GSM8KDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,   # MLX uses unified memory; avoid multiprocessing issues
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        GSM8KDataset(val),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader
