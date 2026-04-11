"""
Data preparation for OpenAI GSM8K math reasoning dataset.

Dataset: openai/gsm8k
Subset: socratic (default)

Dataset format:
- question: str (math problem)
- answer: str (step-by-step solution)
"""

from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset


class GSM8KDataset(Dataset):
    """Dataset for GSM8K math reasoning problems."""
    
    def __init__(self, data, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Format as conversation: user asks question, assistant provides answer
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
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
    seed: int = 42,
    subset: str = "socratic"
) -> Tuple[GSM8KDataset, GSM8KDataset]:
    """
    Load and prepare GSM8K dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length for tokenization
        validation_split: Fraction of data to use for validation
        seed: Random seed for train/val split
        subset: GSM8K subset to use (default: "socratic")
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load dataset from HuggingFace (train split only)
    dataset = load_dataset("openai/gsm8k", subset, split="train")
    
    # Split into train and validation
    val_size = int(len(dataset) * validation_split)
    train_data, val_data = random_split(
        dataset, 
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataset objects
    train_dataset = GSM8KDataset(train_data, tokenizer, max_seq_length)
    val_dataset = GSM8KDataset(val_data, tokenizer, max_seq_length)
    
    return train_dataset, val_dataset
