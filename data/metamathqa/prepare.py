"""Save reproducible MetaMathQA splits with parseable reference answers."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from math_verify import LatexExtractionConfig, parse
from tqdm import tqdm

DATASET = "meta-math/MetaMathQA"
DATASET_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
SOURCE_FILE = "MetaMathQA-395K.json"
ANSWER_MARKER = "The answer is:"


def file_sha256(path):
    """Hash a file without loading its contents into memory."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    """Extract answers, deduplicate, filter unparseable references, and split rows."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory for train.jsonl, test.jsonl, and manifest.json",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.05,
        help="Fraction of retained rows assigned to test, rounded up",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument(
        "--keep-duplicate-queries", action="store_true",
        help="Keep repeated queries; by default keep only the first exact match",
    )
    parser.add_argument(
        "--source-file", type=Path, default=None,
        help="Use an already downloaded source JSON instead of downloading it",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1, exclusive")

    source_path = args.source_file
    if source_path is None:
        source_path = Path(hf_hub_download(
            repo_id=DATASET, repo_type="dataset", revision=DATASET_REVISION,
            filename=SOURCE_FILE,
        ))
    rows = pd.read_json(source_path, dtype=False)
    source_rows = len(rows)
    if not rows["response"].str.contains(ANSWER_MARKER, regex=False).all():
        raise ValueError("Every response must contain 'The answer is:'")

    # Preserve the final suffix verbatim except for surrounding whitespace.
    rows["ground_truth"] = rows["response"].str.rsplit(
        ANSWER_MARKER, n=1,
    ).str[-1].str.strip()
    rows = rows.drop(columns=["original_question"])
    if not args.keep_duplicate_queries:
        rows = rows.drop_duplicates(subset="query", keep="first")
    duplicates_removed = source_rows - len(rows)
    rows_after_deduplication = len(rows)
    empty_ground_truth_removed = int(rows["ground_truth"].eq("").sum())

    # Match evaluation's reference parser; keep raw answer text in the output.
    parseable = [
        bool(answer) and bool(parse(
            f"${answer}$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        ))
        for answer in tqdm(rows["ground_truth"], desc="Parse reference answers", unit="row")
    ]
    rows = rows.loc[parseable]
    unparseable_references_removed = rows_after_deduplication - len(rows)

    dataset = Dataset.from_pandas(rows, preserve_index=False)
    splits = dataset.train_test_split(test_size=args.test_fraction, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name in ("train", "test"):
        path = args.output_dir / f"{name}.jsonl"
        splits[name].to_json(path, force_ascii=False)
        output_hashes[path.name] = file_sha256(path)

    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION if args.source_file is None else None,
        "source_file": str(source_path) if args.source_file else SOURCE_FILE,
        "source_sha256": file_sha256(source_path),
        "source_rows": source_rows,
        "deduplicate_queries": not args.keep_duplicate_queries,
        "deduplication": "Exact query equality; keep first row in source order",
        "duplicates_removed": duplicates_removed,
        "rows_after_deduplication": rows_after_deduplication,
        "reference_filter": {
            "library": "math-verify",
            "input": "ground_truth wrapped in dollar-sign math delimiters",
            "extraction_config": "LatexExtractionConfig()",
            "fallback_mode": "no_fallback",
            "extraction_mode": "any_match",
            "parsing_timeout_seconds": 5,
            "raise_on_error": False,
            "keep_rule": "Nonempty ground_truth and a nonempty parse result",
            "order": "After optional query deduplication, before train/test splitting",
        },
        "unparseable_references_removed": unparseable_references_removed,
        "empty_ground_truth_removed": empty_ground_truth_removed,
        "retained_rows": len(rows),
        "empty_ground_truth": int(rows["ground_truth"].eq("").sum()),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_method": "Hugging Face Dataset.train_test_split; shuffled rows",
        "train_rows": len(splits["train"]),
        "test_rows": len(splits["test"]),
        "sha256": output_hashes,
        "prepare_script_sha256": file_sha256(__file__),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in (
                "datasets", "pandas", "huggingface-hub", "numpy", "pyarrow",
                "math-verify", "latex2sympy2-extended", "sympy", "antlr4-python3-runtime",
            )
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Loaded {source_rows:,} rows; removed {duplicates_removed:,} duplicate queries.")
    print(
        f"Removed {unparseable_references_removed:,} unparseable reference answers "
        f"(including {empty_ground_truth_removed:,} empty answers)."
    )
    print(f"Saved {len(splits['train']):,} train / {len(splits['test']):,} test rows to {args.output_dir}")


if __name__ == "__main__":
    main()
