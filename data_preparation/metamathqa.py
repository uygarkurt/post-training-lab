"""Local MetaMathQA SFT data, evaluation prompts, and answer matching."""

from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data_preparation.sft import build_sft_dataloader, encode_sft_example

TRAIN_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data/metamathqa/train.jsonl"
)
TEST_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data/metamathqa/test.jsonl"
)
ANSWER_MARKER = "The answer is:"


class MetaMathQASFTDataset(Dataset):
    """Tokenize and hold local MetaMathQA query-response examples for SFT."""

    def __init__(self, tokenizer, max_seq_len):
        """Load training rows and retain examples within the token limit."""
        self.samples = []
        self.total_rows = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0

        for row in hf_load_dataset(
            "json", data_files=str(TRAIN_DATASET_PATH), split="train",
        ):
            self.total_rows += 1
            question = row["query"]
            response = row["response"]

            sample = encode_sft_example(tokenizer, question, response)
            if sample is None or len(sample["input_ids"]) < 4:
                self.skipped_invalid += 1
                continue

            if len(sample["input_ids"]) > max_seq_len:
                self.skipped_overlong += 1
                continue

            self.samples.append(sample)

    def __len__(self):
        """Return the number of retained training samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized query-response sample."""
        return self.samples[index]

    def print_summary(self, destination):
        """Print filtering statistics followed by a split description."""
        print(
            f"  {self.total_rows} rows loaded  "
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
        """Build deterministic MetaMathQA train and validation datasets."""
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
        """Build matching tiny MetaMathQA train and validation datasets."""
        dataset = cls(tokenizer, max_seq_len=max_seq_len)

        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        debug_indices = shuffled_indices[: min(debug_samples, len(dataset))]

        train_dataset = Subset(dataset, debug_indices)
        val_dataset = Subset(dataset, debug_indices)

        dataset.print_summary(
            f"debug overfit: {len(train_dataset)} MetaMathQA samples "
            "(same set for train and val)"
        )
        return train_dataset, val_dataset


def parse_reference(answer):
    """Parse a final answer, including bare LaTeX from the source response."""
    from math_verify import LatexExtractionConfig, parse

    if not answer.strip():
        return []
    return parse(
        f"${answer}$",
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
    )


def grade_answer(completion_text, ground_truth):
    """Return the parsed model answer and its mathematical correctness."""
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    if not ground_truth:
        return [], False
    # A source-style final suffix takes precedence over answers in the reasoning.
    if ANSWER_MARKER in completion_text:
        predicted_answer = parse_reference(
            completion_text.rpartition(ANSWER_MARKER)[2].strip(),
        )
    else:
        predicted_answer = parse(
            completion_text,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
        )
    correct = bool(predicted_answer) and verify(ground_truth, predicted_answer)
    return predicted_answer, correct


def _make_eval_collate_fn(tokenizer):
    """Build a collator that left-pads prompts and preserves parsed answers."""
    def collate_fn(batch):
        """Pad one list of evaluation prompts for batched generation."""
        padded_prompts = tokenizer.pad(
            {"input_ids": [sample["prompt_ids"] for sample in batch]},
            padding=True,
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "prompt_ids": padded_prompts["input_ids"],
            "prompt_attention_mask": padded_prompts["attention_mask"],
            "ground_truth": [sample["ground_truth"] for sample in batch],
            "question": [sample["question"] for sample in batch],
            "answer": [sample["answer"] for sample in batch],
        }

    return collate_fn


def build_eval_dataloader(dataset, tokenizer, batch_size):
    """Build a deterministic DataLoader of padded evaluation prompts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_eval_collate_fn(tokenizer),
        num_workers=0,
        drop_last=False,
    )


class MetaMathQAEvalDataset(Dataset):
    """Tokenize local test queries and parse their extracted reference answers."""

    def __init__(
        self, tokenizer, max_prompt_len, num_samples=None,
        dataset_path=TEST_DATASET_PATH,
    ):
        """Load test rows, optionally retaining only the first N usable samples."""
        self.samples = []
        self.skipped = 0
        self.skipped_overlong = 0
        self.unparsed_answers = 0
        self.empty_queries = 0

        for row in hf_load_dataset(
            "json", data_files={"test": str(dataset_path)}, split="test",
        ):
            question = row["query"]
            if not question.strip():
                self.empty_queries += 1
                self.skipped += 1
                continue
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt_text)
            if len(prompt_ids) > max_prompt_len:
                self.skipped_overlong += 1
                self.skipped += 1
                continue

            ground_truth = parse_reference(row["ground_truth"])
            if not ground_truth:
                self.unparsed_answers += 1
                self.skipped += 1
                continue
            self.samples.append({
                "prompt_ids": prompt_ids,
                "ground_truth": ground_truth,
                "question": question,
                "answer": row["ground_truth"],
            })

        if num_samples is not None:
            self.samples = self.samples[:num_samples]

    def __len__(self):
        """Return the number of retained test samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized query and its reference answer."""
        return self.samples[index]
