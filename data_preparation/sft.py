"""Shared PyTorch padding and batching for all SFT datasets."""

import torch
from torch.utils.data import DataLoader


def encode_sft_example(tokenizer, user_content, assistant_content):
    """Tokenize one chat with an exact generation-prompt prefix."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    if not full_text.startswith(prompt_text):
        return None

    prompt_ids = tokenizer.encode(prompt_text)
    assistant_ids = tokenizer.encode(
        full_text[len(prompt_text):],
        add_special_tokens=False,
    )
    if not prompt_ids or not assistant_ids:
        return None

    return {
        "input_ids": prompt_ids + assistant_ids,
        "loss_mask": [0] * len(prompt_ids) + [1] * len(assistant_ids),
    }


def _make_sft_collate_fn(tokenizer):
    """Build a collator that right-pads SFT samples and their loss masks."""
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
