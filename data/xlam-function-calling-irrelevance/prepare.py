"""Prepare xLAM calls and irrelevant-tool abstentions for SFT."""

import argparse
import importlib.util
import json
import math
import platform
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import hf_hub_download


XLAM_PREPARE_PATH = (
    Path(__file__).resolve().parents[1] / "xlam-function-calling-60k/prepare.py"
)
XLAM_PREPARE_SPEC = importlib.util.spec_from_file_location("xlam_prepare", XLAM_PREPARE_PATH)
xlam_prepare = importlib.util.module_from_spec(XLAM_PREPARE_SPEC)
XLAM_PREPARE_SPEC.loader.exec_module(xlam_prepare)

IRRELEVANCE_DATASET = "MadeAgents/xlam-irrelevance-7.5k"
IRRELEVANCE_REVISION = "34323bf09efc7e4a394998a0fa91ff997617c369"
IRRELEVANCE_SOURCE_FILE = "xlam-7.5k-irrelevancek.json"


def query_key(row):
    """Match source examples with the same whitespace-normalized query."""
    return " ".join(row["query"].split())


def normalize_irrelevance_row(row, index):
    """Normalize a no-call row, excluding rows without offered tools."""
    if not isinstance(row, dict):
        raise ValueError("irrelevance row must be an object")
    query = row["query"]
    tools = json.loads(row["tools"])
    answers = json.loads(row["answers"])
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if not isinstance(tools, list) or answers != []:
        raise ValueError("irrelevance rows need a tool list and no calls")
    if not tools:
        return None

    normalized_tools = [xlam_prepare.normalize_tool(tool) for tool in tools]
    tool_names = [tool["function"]["name"] for tool in normalized_tools]
    if len(tool_names) != len(set(tool_names)):
        raise xlam_prepare.DuplicateToolNameError(
            "available tool names must be unique"
        )
    return {
        "id": f"irrelevance:{index}",
        "query": query,
        "tools": json.dumps(
            normalized_tools, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ),
        "answers": "[]",
    }


def main():
    """Download both sources, normalize them, and save mixed JSONL splits."""
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
        help="Target held-out fraction of each source",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and shuffle seed")
    parser.add_argument(
        "--xlam-source-file", type=Path, default=None,
        help="Use a local original xLAM JSON file instead of downloading it",
    )
    parser.add_argument(
        "--irrelevance-source-file", type=Path, default=None,
        help="Use a local Hammer irrelevance JSON file instead of downloading it",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1, exclusive")

    xlam_source_path = args.xlam_source_file
    if xlam_source_path is None:
        xlam_source_path = Path(hf_hub_download(
            repo_id=xlam_prepare.DATASET,
            repo_type="dataset",
            revision=xlam_prepare.DATASET_REVISION,
            filename=xlam_prepare.SOURCE_FILE,
        ))
    irrelevance_source_path = args.irrelevance_source_file
    if irrelevance_source_path is None:
        irrelevance_source_path = Path(hf_hub_download(
            repo_id=IRRELEVANCE_DATASET,
            repo_type="dataset",
            revision=IRRELEVANCE_REVISION,
            filename=IRRELEVANCE_SOURCE_FILE,
        ))

    with xlam_source_path.open(encoding="utf-8") as source:
        xlam_source_rows = json.load(source)
    with irrelevance_source_path.open(encoding="utf-8") as source:
        irrelevance_source_rows = json.load(source)
    if not isinstance(xlam_source_rows, list) or not isinstance(irrelevance_source_rows, list):
        raise ValueError("Both source files must contain one JSON array")

    xlam_rows = []
    irrelevance_rows = []
    no_tool_rows_excluded = 0
    invalid_counts = defaultdict(int)
    for row in xlam_source_rows:
        try:
            normalized = xlam_prepare.normalize_row(row)
        except xlam_prepare.DuplicateToolNameError:
            invalid_counts["xlam_duplicate_tool_names"] += 1
            continue
        except (KeyError, TypeError, ValueError):
            invalid_counts["xlam_other_invalid"] += 1
            continue
        normalized["id"] = f"xlam:{normalized['id']}"
        xlam_rows.append(normalized)

    for index, row in enumerate(irrelevance_source_rows):
        try:
            normalized = normalize_irrelevance_row(row, index)
        except xlam_prepare.DuplicateToolNameError:
            invalid_counts["irrelevance_duplicate_tool_names"] += 1
            continue
        except (KeyError, TypeError, ValueError):
            invalid_counts["irrelevance_other_invalid"] += 1
            continue
        if normalized is None:
            no_tool_rows_excluded += 1
            continue
        irrelevance_rows.append(normalized)

    if not xlam_rows or not irrelevance_rows:
        raise ValueError("Both sources must retain at least one valid row")

    xlam_train, xlam_test, xlam_query_groups = xlam_prepare.split_rows(
        xlam_rows, test_fraction=args.test_fraction, seed=args.seed,
    )
    train_queries = {query_key(row) for row in xlam_train}
    test_queries = {query_key(row) for row in xlam_test}
    irrelevance_train, irrelevance_test, unmatched_groups = [], [], defaultdict(list)
    for row in irrelevance_rows:
        key = query_key(row)
        if key in test_queries:
            irrelevance_test.append(row)
        elif key in train_queries:
            irrelevance_train.append(row)
        else:
            unmatched_groups[key].append(row)

    # Rows derived from the same query must stay on the same side of the split.
    # The negative fraction can differ slightly from 5% when queries overlap.
    negative_test_target = math.ceil(len(irrelevance_rows) * args.test_fraction)
    generator = random.Random(args.seed)
    groups = list(unmatched_groups.values())
    generator.shuffle(groups)
    for group in groups:
        if len(irrelevance_test) + len(group) <= negative_test_target:
            irrelevance_test.extend(group)
        else:
            irrelevance_train.extend(group)

    train_rows = xlam_train + irrelevance_train
    test_rows = xlam_test + irrelevance_test
    if {query_key(row) for row in train_rows} & {query_key(row) for row in test_rows}:
        raise ValueError("A query appears in both prepared splits")
    generator.shuffle(train_rows)
    generator.shuffle(test_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, rows in (("train", train_rows), ("test", test_rows)):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ) + "\n")
        output_hashes[path.name] = xlam_prepare.file_sha256(path)

    manifest = {
        "sources": {
            "xlam": {
                "dataset": xlam_prepare.DATASET,
                "revision": (
                    xlam_prepare.DATASET_REVISION
                    if args.xlam_source_file is None else None
                ),
                "source_file": (
                    str(xlam_source_path)
                    if args.xlam_source_file else xlam_prepare.SOURCE_FILE
                ),
                "source_sha256": xlam_prepare.file_sha256(xlam_source_path),
                "source_rows": len(xlam_source_rows),
                "retained_rows": len(xlam_rows),
                "train_rows": len(xlam_train),
                "test_rows": len(xlam_test),
            },
            "irrelevance": {
                "dataset": IRRELEVANCE_DATASET,
                "revision": IRRELEVANCE_REVISION if args.irrelevance_source_file is None else None,
                "source_file": (
                    str(irrelevance_source_path)
                    if args.irrelevance_source_file else IRRELEVANCE_SOURCE_FILE
                ),
                "source_sha256": xlam_prepare.file_sha256(irrelevance_source_path),
                "source_rows": len(irrelevance_source_rows),
                "no_tool_rows_excluded": no_tool_rows_excluded,
                "retained_rows": len(irrelevance_rows),
                "train_rows": len(irrelevance_train),
                "test_rows": len(irrelevance_test),
                "unmatched_query_groups": len(unmatched_groups),
            },
        },
        "invalid_rows_removed": dict(invalid_counts),
        "split_method": (
            "Exclude irrelevance rows with no offered tools; split xLAM by "
            "whitespace-normalized query as in its original preparation; "
            "assign irrelevance rows with matching queries to the same split; "
            "split unmatched irrelevance query groups toward their 5% target; "
            "shuffle each mixed split"
        ),
        "seed": args.seed,
        "test_fraction_target": args.test_fraction,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "sha256": output_hashes,
        "prepare_script_sha256": xlam_prepare.file_sha256(__file__),
        "xlam_prepare_script_sha256": xlam_prepare.file_sha256(XLAM_PREPARE_PATH),
        "python_version": platform.python_version(),
        "package_versions": {"huggingface-hub": version("huggingface-hub")},
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")

    print(
        f"xLAM: {len(xlam_source_rows):,} loaded, {len(xlam_rows):,} retained, "
        f"{len(xlam_train):,} train / {len(xlam_test):,} test "
        f"({xlam_query_groups:,} query groups)."
    )
    print(
        f"Irrelevance: {len(irrelevance_source_rows):,} loaded, "
        f"{no_tool_rows_excluded:,} no-tool rows excluded, "
        f"{len(irrelevance_rows):,} retained, "
        f"{len(irrelevance_train):,} train / {len(irrelevance_test):,} test."
    )
    print(
        f"Saved {len(train_rows):,} mixed train / {len(test_rows):,} "
        f"mixed test rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
