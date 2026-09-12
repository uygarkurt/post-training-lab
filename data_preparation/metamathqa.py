"""Local MetaMathQA evaluation prompts and mathematical answer matching."""

from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset

# MetaMathQA uses the same padded evaluation batch contract as NuminaMath.
from data_preparation.numinamath import build_eval_dataloader

TEST_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data/metamathqa/test.jsonl"
)
ANSWER_MARKER = "The answer is:"


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
