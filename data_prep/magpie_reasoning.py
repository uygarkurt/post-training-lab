"""
Data preparation for Magpie Reasoning V2 CoT dataset.

Dataset: Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B
Columns: instruction, response

Note: <think> tokens are already present in responses.
Training samples exceeding max_seq_length are discarded (not truncated)
to preserve the complete answer at the end.
"""

from typing import Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset, load_from_disk
import logging


class MagpieReasoningDataset(Dataset):
    """Dataset for Magpie Reasoning V2 CoT problems."""
    
    def __init__(self, data, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Format as conversation: user provides instruction, assistant provides response
        # Note: <think> tokens are already in response
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
        
        # Apply chat template and tokenize
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encodings = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding tokens
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
        }


def prepare_data(
    tokenizer,
    max_seq_length: int = 2048,
    validation_split: float = 0.05,
    seed: int = 42
) -> Tuple[MagpieReasoningDataset, MagpieReasoningDataset]:
    """
    Load and prepare Magpie Reasoning V2 CoT dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length for tokenization
        validation_split: Fraction of data to use for validation
        seed: Random seed for train/val split
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Cache directory
    cache_dir = Path.home() / ".cache" / "post-training-lab" / "filtered_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache path based on dataset and max_seq_length
    cache_name = f"magpie_reasoning_v2_250k_cot_maxlen{max_seq_length}"
    cache_path = cache_dir / cache_name
    
    # Check if filtered dataset is already cached
    if cache_path.exists():
        logging.info(f"Loading cached filtered dataset from {cache_path}")
        dataset = load_from_disk(str(cache_path))
        logging.info(f"Loaded {len(dataset)} samples from cache")
    else:
        # Load dataset from HuggingFace (train split)
        logging.info("Loading dataset from HuggingFace...")
        dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", split="train")
        
        # Filter out samples that exceed max_seq_length (parallelized)
        logging.info(f"Filtering samples by max_seq_length={max_seq_length}...")
        original_size = len(dataset)
        
        def check_length(example):
            """Check if example fits within max_seq_length."""
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens) <= max_seq_length
        
        # Use HuggingFace's parallelized filter
        dataset = dataset.filter(check_length, num_proc=4, desc="Filtering by length")
        
        # Log filtering results
        filtered_size = len(dataset)
        discarded = original_size - filtered_size
        discard_percentage = (discarded / original_size * 100) if original_size > 0 else 0
        logging.info(
            f"Dataset filtering: kept {filtered_size}/{original_size} samples "
            f"({discard_percentage:.1f}% discarded)"
        )
        
        # Save filtered dataset to cache
        logging.info(f"Saving filtered dataset to {cache_path}")
        dataset.save_to_disk(str(cache_path))
    
    # Split into train and validation
    val_size = int(len(dataset) * validation_split)
    train_data, val_data = random_split(
        dataset, 
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataset objects
    train_dataset = MagpieReasoningDataset(train_data, tokenizer, max_seq_length)
    val_dataset = MagpieReasoningDataset(val_data, tokenizer, max_seq_length)
    
    return train_dataset, val_dataset
