"""Local xLAM function-calling SFT/GRPO data, rewards, and evaluation."""

import json
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data_preparation.sft import build_sft_dataloader

TRAIN_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/xlam-function-calling-60k/train.jsonl"
)
TEST_DATASET_PATH = TRAIN_DATASET_PATH.with_name("test.jsonl")
MIXED_TRAIN_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/xlam-function-calling-irrelevance/train.jsonl"
)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class InvalidPreparedRowError(ValueError):
    """Identify malformed rows in a prepared xLAM JSONL file."""


class ToolTemplateError(ValueError):
    """Identify tokenizers whose chat template does not expose tools."""


def _assistant_message(answers):
    """Convert reference calls to the standard assistant message structure."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": answer} for answer in answers
        ],
    }


def parse_prepared_row(row, allow_empty_answers=False, allow_empty_tools=False):
    """Decode one prepared row, optionally allowing no calls or no tools."""
    query = row["query"]
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = (
        json.loads(row["answers"])
        if isinstance(row["answers"], str)
        else row["answers"]
    )
    if not isinstance(query, str) or not query.strip():
        raise InvalidPreparedRowError("query must be a nonempty string")
    if not isinstance(tools, list) or (not tools and not allow_empty_tools):
        raise InvalidPreparedRowError("tools must be a nonempty list")
    if not isinstance(answers, list) or (not answers and not allow_empty_answers):
        raise InvalidPreparedRowError("answers must be a nonempty list")
    if not tools and answers:
        raise InvalidPreparedRowError("a row without tools cannot contain a call")
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("function"), dict)
            or not isinstance(tool["function"].get("name"), str)
        ):
            raise InvalidPreparedRowError(
                "tools must use the standard function schema"
            )
    for answer in answers:
        if (
            not isinstance(answer, dict)
            or not isinstance(answer.get("name"), str)
            or not isinstance(answer.get("arguments"), dict)
        ):
            raise InvalidPreparedRowError(
                "answers must contain names and argument objects"
            )
    return query, tools, answers


def render_prompt(tokenizer, query, tools):
    """Render one tool-aware user prompt with the tokenizer's chat template."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if tools and tools[0]["function"]["name"] not in prompt_text:
        raise ToolTemplateError(
            "The tokenizer chat template did not render the provided tools"
        )
    return prompt_text


def encode_sft_example(tokenizer, query, tools, answers):
    """Tokenize a call or no-call chat and mask its prompt tokens from SFT loss."""
    prompt_text = render_prompt(tokenizer, query, tools)
    assistant_message = (
        _assistant_message(answers)
        if answers else {"role": "assistant", "content": "[]"}
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": query},
            assistant_message,
        ],
        tools=tools,
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


class XLAMFunctionCallingSFTDataset(Dataset):
    """Tokenize and hold local xLAM function-calling examples for SFT."""

    def __init__(self, tokenizer, max_seq_len, dataset_path=TRAIN_DATASET_PATH):
        """Load training rows and retain complete examples within the limit."""
        self.samples = []
        self.total_rows = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0
        self.no_call_rows = 0
        self.retained_no_call = 0
        self.skipped_no_call_overlong = 0

        for row in hf_load_dataset(
            "json", data_files=str(dataset_path), split="train",
        ):
            self.total_rows += 1
            try:
                query, tools, answers = parse_prepared_row(
                    row, allow_empty_answers=True, allow_empty_tools=True,
                )
                if not answers:
                    self.no_call_rows += 1
                sample = encode_sft_example(
                    tokenizer,
                    query,
                    tools,
                    answers,
                )
            except (InvalidPreparedRowError, KeyError, json.JSONDecodeError):
                sample = None
            if sample is None or len(sample["input_ids"]) < 4:
                self.skipped_invalid += 1
                continue
            if len(sample["input_ids"]) > max_seq_len:
                self.skipped_overlong += 1
                if not answers:
                    self.skipped_no_call_overlong += 1
                continue
            self.samples.append(sample)
            if not answers:
                self.retained_no_call += 1

    def __len__(self):
        """Return the number of retained training samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized function-calling sample."""
        return self.samples[index]

    def print_summary(self, destination):
        """Print filtering statistics followed by a split description."""
        print(
            f"  {self.total_rows} rows loaded  "
            f"→  {len(self)} retained / {self.skipped_overlong} overlong / "
            f"{self.skipped_invalid} invalid  →  {destination}."
        )
        if self.no_call_rows:
            print(
                f"  No-call rows: {self.retained_no_call}/{self.no_call_rows} retained; "
                f"{self.skipped_no_call_overlong} overlong."
            )

    @classmethod
    def build_train_val_datasets(
        cls,
        tokenizer,
        max_seq_len,
        val_split,
        seed,
        dataset_path=TRAIN_DATASET_PATH,
    ):
        """Build deterministic xLAM training and validation datasets."""
        dataset = cls(
            tokenizer, max_seq_len=max_seq_len, dataset_path=dataset_path,
        )
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
        dataset_path=TRAIN_DATASET_PATH,
    ):
        """Build matching tiny xLAM train and validation datasets."""
        dataset = cls(
            tokenizer, max_seq_len=max_seq_len, dataset_path=dataset_path,
        )
        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        debug_indices = shuffled_indices[:min(debug_samples, len(dataset))]
        train_dataset = Subset(dataset, debug_indices)
        val_dataset = Subset(dataset, debug_indices)
        dataset.print_summary(
            f"debug overfit: {len(train_dataset)} xLAM samples "
            "(same set for train and val)"
        )
        return train_dataset, val_dataset


class XLAMFunctionCallingGRPODataset(Dataset):
    """Tokenize and hold local xLAM prompts and reference calls for GRPO."""

    def __init__(self, tokenizer, max_prompt_len):
        """Load training rows and retain complete prompts within the limit."""
        self.samples = []
        self.total_rows = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0

        for row in hf_load_dataset(
            "json", data_files=str(TRAIN_DATASET_PATH), split="train",
        ):
            self.total_rows += 1
            try:
                query, tools, answers = parse_prepared_row(row)
                prompt_text = render_prompt(tokenizer, query, tools)
                prompt_ids = tokenizer.encode(prompt_text)
            except (InvalidPreparedRowError, KeyError, json.JSONDecodeError):
                self.skipped_invalid += 1
                continue
            if len(prompt_ids) > max_prompt_len:
                self.skipped_overlong += 1
                continue
            self.samples.append({
                "prompt_ids": prompt_ids,
                "id": row["id"],
                "query": query,
                "tools": tools,
                "answers": answers,
            })

    def __len__(self):
        """Return the number of retained training samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized prompt and its reference tool calls."""
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
        max_prompt_len,
        val_split,
        seed,
    ):
        """Build deterministic xLAM GRPO training and validation datasets."""
        dataset = cls(tokenizer, max_prompt_len=max_prompt_len)
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
        max_prompt_len,
        seed,
        debug_samples,
    ):
        """Build matching tiny xLAM GRPO train and validation datasets."""
        dataset = cls(tokenizer, max_prompt_len=max_prompt_len)
        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        debug_indices = shuffled_indices[:min(debug_samples, len(dataset))]
        train_dataset = Subset(dataset, debug_indices)
        val_dataset = Subset(dataset, debug_indices)
        dataset.print_summary(
            f"debug overfit: {len(train_dataset)} xLAM samples "
            "(same set for train and val)"
        )
        return train_dataset, val_dataset


def _normalize_predicted_call(call):
    """Normalize one common tool-call object or reject it as malformed."""
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("function"), dict):
        call = call["function"]
    name = call.get("name")
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def parse_tool_calls(completion_text):
    """Extract tool calls from tagged Qwen output or a bare JSON response."""
    tagged_calls = TOOL_CALL_PATTERN.findall(completion_text)
    if tagged_calls:
        raw_calls = []
        for payload in tagged_calls:
            try:
                raw_calls.append(json.loads(payload))
            except json.JSONDecodeError:
                return []
    else:
        payload = completion_text.strip()
        code_fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.DOTALL | re.IGNORECASE,
        )
        if code_fence:
            payload = code_fence.group(1)
        try:
            raw_calls = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list):
            return []

    normalized_calls = [_normalize_predicted_call(call) for call in raw_calls]
    if any(call is None for call in normalized_calls):
        return []
    return normalized_calls


def _canonical_call(call):
    """Serialize a call canonically while preserving JSON value types."""
    return json.dumps(
        call,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def grade_tool_calls(completion_text, reference_calls):
    """Grade unordered function names and complete calls for one completion."""
    predicted_calls = parse_tool_calls(completion_text)
    if not reference_calls:
        payload = completion_text.strip()
        code_fence = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", payload,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if code_fence:
            payload = code_fence.group(1)
        no_call = (
            not predicted_calls
            and "<tool_call" not in completion_text
            and (not payload.startswith(("{", "[")) or payload == "[]")
        )
        return predicted_calls, no_call, no_call
    predicted_names = Counter(call["name"] for call in predicted_calls)
    reference_names = Counter(call["name"] for call in reference_calls)
    names_correct = bool(predicted_calls) and predicted_names == reference_names
    calls_correct = bool(predicted_calls) and Counter(
        _canonical_call(call) for call in predicted_calls
    ) == Counter(_canonical_call(call) for call in reference_calls)
    return predicted_calls, names_correct, calls_correct


def tool_call_rewards(rollouts_text, reference_calls):
    """Return binary exact-call-set rewards for decoded completions."""
    return [
        float(grade_tool_calls(text, reference_calls)[2])
        for text in rollouts_text
    ]


def _make_eval_collate_fn(tokenizer):
    """Build a collator that left-pads prompts and preserves call references."""
    def collate_fn(batch):
        """Pad one list of tool-aware evaluation prompts."""
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
            "id": [sample["id"] for sample in batch],
            "query": [sample["query"] for sample in batch],
            "tools": [sample["tools"] for sample in batch],
            "answers": [sample["answers"] for sample in batch],
        }

    return collate_fn


def build_eval_dataloader(dataset, tokenizer, batch_size):
    """Build a deterministic DataLoader of padded tool-aware prompts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_eval_collate_fn(tokenizer),
        num_workers=0,
        drop_last=False,
    )


class XLAMFunctionCallingEvalDataset(Dataset):
    """Tokenize local xLAM test prompts with their available tools."""

    def __init__(
        self,
        tokenizer,
        max_prompt_len,
        num_samples=None,
        dataset_path=TEST_DATASET_PATH,
    ):
        """Load test rows, optionally retaining only the first N usable rows."""
        self.samples = []
        self.skipped = 0
        self.skipped_overlong = 0
        self.skipped_invalid = 0

        for row in hf_load_dataset(
            "json", data_files={"test": str(dataset_path)}, split="test",
        ):
            try:
                query, tools, answers = parse_prepared_row(
                    row, allow_empty_answers=True, allow_empty_tools=True,
                )
                prompt_text = render_prompt(tokenizer, query, tools)
                prompt_ids = tokenizer.encode(prompt_text)
            except (InvalidPreparedRowError, KeyError, json.JSONDecodeError):
                self.skipped_invalid += 1
                self.skipped += 1
                continue
            if len(prompt_ids) > max_prompt_len:
                self.skipped_overlong += 1
                self.skipped += 1
                continue
            self.samples.append({
                "prompt_ids": prompt_ids,
                "id": row["id"],
                "query": query,
                "tools": tools,
                "answers": answers,
            })

        if num_samples is not None:
            self.samples = self.samples[:num_samples]

    def __len__(self):
        """Return the number of retained test samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one tokenized prompt and its reference tool calls."""
        return self.samples[index]
