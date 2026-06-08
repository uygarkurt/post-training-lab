"""
Data loading and batching for
Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B.

Each sample is a chat-formatted (input_ids, loss_mask) pair where:
    loss_mask = 0  → prompt token   (no gradient)
    loss_mask = 1  → response token (supervised)

Tokenisation is lazy (happens in __getitem__) so startup is near-instant
even for large datasets.

Public API
----------
build_dataloaders(tokenizer, args) -> (train_loader, val_loader)
    Returns two PyTorch DataLoaders whose batches are (torch.Tensor, torch.Tensor)
    tuples of shape [B, T] ready to be converted to the target accelerator format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset


DATASET_NAME  = "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B"
DATASET_SPLIT = "train"


# ---------------------------------------------------------------------------
# Dataset — lazy tokenisation
# ---------------------------------------------------------------------------

class MagpieReasoningDataset(Dataset):
    """
    Wraps a HuggingFace dataset slice and tokenises on-the-fly in __getitem__.
    No upfront tokenisation loop — startup is O(1) in dataset size.
    """

    def __init__(self, hf_ds, tokenizer, max_seq_len):
        self.ds          = hf_ds
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row         = self.ds[idx]
        instruction = row["instruction"]
        response    = row["response"]

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer.encode(prompt_text)
        full_ids   = self.tokenizer.encode(full_text)[:self.max_seq_len]

        prompt_len = min(len(prompt_ids), len(full_ids)) # In case the prompt itself exceeds max_seq_len, we treat all tokens as prompt. In that case, the loss_mask will be all zeros, and the collate_fn will filter out this sample from the batch.
        loss_mask  = [0] * prompt_len + [1] * (len(full_ids) - prompt_len) # Response tokens get a loss mask of 1, prompt tokens get 0.

        return full_ids, loss_mask


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _make_collate_fn(pad_id):
    """
    Right-pads sequences within a batch to the same length and converts to
    torch.Tensor arrays. Filters out any samples where no response token survived
    truncation (loss_mask is all zeros). Receives a batch 
    of (input_ids, loss_mask) pairs
    """
    def collate_fn(batch):
        # Drop samples whose response was entirely truncated away.
        batch = [(ids, msk) for ids, msk in batch if sum(msk) > 0]
        if not batch:
            # Return a minimal placeholder batch; the loss will be ~0.
            batch = [([pad_id, pad_id], [0, 0])]

        batch_ids = [item[0] for item in batch]
        batch_msk = [item[1] for item in batch]
        max_len   = max(len(ids) for ids in batch_ids)

        # Right-pad so that all sequences in the batch have the same length. 
        padded_ids, padded_masks = [], []
        for ids, mask in zip(batch_ids, batch_msk):
            pad = max_len - len(ids)
            padded_ids.append(ids   + [pad_id] * pad)
            padded_masks.append(mask + [0]     * pad)

        return (
            torch.tensor(padded_ids,   dtype=torch.int32),
            torch.tensor(padded_masks, dtype=torch.float32),
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_dataloaders(tokenizer, args):
    """
    Load and split the dataset, return two DataLoaders.

    The HF dataset is shuffled and split using its native fast ops (Arrow
    backed, no Python loop). Tokenisation is deferred to batch time.

    Each batch yielded by the loaders is a tuple:
        (input_ids: torch.Tensor [B, T], loss_mask: torch.Tensor [B, T])
    """
    ds = hf_load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Fast pre-filter: drop rows with missing/empty text (Arrow-backed, fast).
    ds = ds.filter(
        lambda batch: [
            bool(inst and resp)
            for inst, resp in zip(batch["instruction"], batch["response"])
        ],
        batched=True,
        desc="Filtering empty rows",
    )

    # Shuffle + split entirely within HF dataset (no tokenisation yet).
    ds = ds.shuffle(seed=args.seed)
    n_val    = max(1, int(len(ds) * args.val_split))
    val_ds   = ds.select(range(n_val))
    train_ds = ds.select(range(n_val, len(ds)))

    print(f"  {len(ds)} samples  →  {len(train_ds)} train / {len(val_ds)} val  "
          f"({args.val_split * 100:.0f}% val split).")

    collate = _make_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        MagpieReasoningDataset(train_ds, tokenizer, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,   # MLX unified memory; avoid multiprocessing issues
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        MagpieReasoningDataset(val_ds, tokenizer, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader

