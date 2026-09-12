"""Normalize xLAM tool calls and save reproducible local train/test splits."""

import argparse
import hashlib
import json
import math
import platform
import random
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "lockon/xlam-function-calling-60k"
DATASET_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
SOURCE_FILE = "xlam_function_calling_60k.json"


class DuplicateToolNameError(ValueError):
    """Identify a row whose available functions do not have unique names."""


def file_sha256(path):
    """Hash a file without loading its contents into memory."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def split_generic_arguments(arguments):
    """Split comma-separated generic arguments without splitting nested types."""
    parts = []
    start = 0
    depth = 0
    for index, character in enumerate(arguments):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(arguments[start:index].strip())
            start = index + 1
    parts.append(arguments[start:].strip())
    return [part for part in parts if part]


def source_type_expression(source_type):
    """Remove optional/default suffixes without cutting commas inside generics."""
    return split_generic_arguments(source_type)[0]


def json_schema_for_type(source_type):
    """Map one Python-like xLAM type expression to a JSON Schema fragment."""
    expression = source_type_expression(source_type)
    lower_expression = expression.lower()
    if lower_expression.startswith("list[") and expression.endswith("]"):
        item_type = expression[5:-1]
        return {"type": "array", "items": json_schema_for_type(item_type)}
    if lower_expression.startswith("tuple[") and expression.endswith("]"):
        tuple_types = split_generic_arguments(expression[6:-1])
        return {
            "type": "array",
            "prefixItems": [json_schema_for_type(item) for item in tuple_types],
            "minItems": len(tuple_types),
            "maxItems": len(tuple_types),
        }
    if lower_expression.startswith("union[") and expression.endswith("]"):
        union_types = split_generic_arguments(expression[6:-1])
        return {"anyOf": [json_schema_for_type(item) for item in union_types]}
    if lower_expression.startswith("str"):
        return {"type": "string"}
    if lower_expression.startswith("int"):
        return {"type": "integer"}
    if lower_expression.startswith("float"):
        return {"type": "number"}
    if lower_expression.startswith("bool"):
        return {"type": "boolean"}
    if lower_expression.startswith(("list", "tuple", "set")):
        return {"type": "array"}
    if lower_expression.startswith("dict"):
        return {"type": "object"}
    # Callable arguments in the source are represented as strings in JSON.
    if lower_expression.startswith("callable"):
        return {"type": "string"}
    return {"type": "string"}


def normalize_tool(tool):
    """Convert one xLAM tool definition to Transformers' tool schema."""
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
        raise ValueError("tool must contain a string name")
    if not tool["name"].strip() or not isinstance(tool.get("parameters"), dict):
        raise ValueError("tool name must be nonempty and parameters must be an object")

    properties = {}
    required = []
    for parameter_name, parameter in tool["parameters"].items():
        if not isinstance(parameter_name, str) or not isinstance(parameter, dict):
            raise ValueError("tool parameters must map string names to objects")
        source_type = parameter.get("type", "str")
        if not isinstance(source_type, str):
            raise ValueError("tool parameter type must be a string")

        property_schema = json_schema_for_type(source_type)
        description = parameter.get("description")
        if isinstance(description, str) and description:
            property_schema["description"] = description
        if "default" in parameter:
            property_schema["default"] = parameter["default"]
        properties[parameter_name] = property_schema

        optional = "optional" in source_type.lower() or "default" in parameter
        if not optional:
            required.append(parameter_name)

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    function = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": parameters,
    }
    return {"type": "function", "function": function}


def normalize_row(row):
    """Parse and validate one source row, returning its local representation."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), int):
        raise ValueError("row must contain an integer id")
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")

    tools = json.loads(row["tools"])
    answers = json.loads(row["answers"])
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must decode to a nonempty list")
    if not isinstance(answers, list) or not answers:
        raise ValueError("answers must decode to a nonempty list")

    normalized_tools = [normalize_tool(tool) for tool in tools]
    tool_names = [tool["function"]["name"] for tool in normalized_tools]
    if len(tool_names) != len(set(tool_names)):
        raise DuplicateToolNameError("available tool names must be unique")
    available_tool_names = set(tool_names)
    available_parameters = {}
    for tool in normalized_tools:
        function = tool["function"]
        available_parameters.setdefault(function["name"], set()).update(
            function["parameters"]["properties"]
        )
    for answer in answers:
        if (
            not isinstance(answer, dict)
            or not isinstance(answer.get("name"), str)
            or not answer["name"].strip()
            or not isinstance(answer.get("arguments"), dict)
        ):
            raise ValueError("each answer must contain a name and argument object")
        if answer["name"] not in available_tool_names:
            raise ValueError("answer names a tool that is not available")
        if not set(answer["arguments"]).issubset(
            available_parameters[answer["name"]]
        ):
            raise ValueError("answer contains an argument absent from its tool")

    return {
        "id": row["id"],
        "query": query,
        # Keep nested, heterogeneous objects encoded so Hugging Face Datasets
        # does not infer one enormous struct from every parameter name.
        "tools": json.dumps(
            normalized_tools, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ),
        "answers": json.dumps(
            answers, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ),
    }


def split_rows(rows, test_fraction, seed):
    """Shuffle and split rows without separating whitespace-equal queries."""
    query_groups = {}
    for row in rows:
        query_key = " ".join(row["query"].split())
        query_groups.setdefault(query_key, []).append(row)

    groups = list(query_groups.values())
    generator = random.Random(seed)
    generator.shuffle(groups)
    target_test_rows = math.ceil(len(rows) * test_fraction)
    train_rows, test_rows = [], []
    for group in groups:
        if len(test_rows) + len(group) <= target_test_rows:
            test_rows.extend(group)
        else:
            train_rows.extend(group)
    if len(test_rows) != target_test_rows:
        raise ValueError("Cannot reach the test size without splitting duplicate queries")

    generator.shuffle(train_rows)
    generator.shuffle(test_rows)
    return train_rows, test_rows, len(query_groups)


def main():
    """Download, normalize, validate, split, and save the xLAM dataset."""
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
        "--source-file", type=Path, default=None,
        help="Use an already downloaded source JSON instead of downloading it",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1, exclusive")

    source_path = args.source_file
    if source_path is None:
        source_path = Path(hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=SOURCE_FILE,
        ))
    with source_path.open(encoding="utf-8") as source:
        source_rows = json.load(source)
    if not isinstance(source_rows, list):
        raise ValueError("The xLAM source file must contain one JSON array")

    rows = []
    duplicate_tool_name_rows = 0
    other_invalid_rows = 0
    for row in source_rows:
        try:
            rows.append(normalize_row(row))
        except DuplicateToolNameError:
            duplicate_tool_name_rows += 1
        except (KeyError, TypeError, ValueError):
            other_invalid_rows += 1
    if not rows:
        raise ValueError("No valid xLAM rows remain after normalization")
    invalid_rows = duplicate_tool_name_rows + other_invalid_rows

    train_rows, test_rows, unique_queries = split_rows(
        rows,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, split in (("train", train_rows), ("test", test_rows)):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in split:
                output.write(json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ) + "\n")
        output_hashes[path.name] = file_sha256(path)

    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION if args.source_file is None else None,
        "source_file": str(source_path) if args.source_file else SOURCE_FILE,
        "source_sha256": file_sha256(source_path),
        "source_rows": len(source_rows),
        "invalid_rows_removed": invalid_rows,
        "invalid_row_reasons": {
            "duplicate_available_tool_name": duplicate_tool_name_rows,
            "other_schema_or_reference_error": other_invalid_rows,
        },
        "retained_rows": len(rows),
        "unique_whitespace_normalized_queries": unique_queries,
        "tool_normalization": {
            "input": "JSON-encoded xLAM tools and answers strings",
            "output": "JSON-encoded Transformers tool schemas and answer-call objects",
            "required_rule": "Parameter type lacks 'optional' and parameter has no default",
            "type_mapping": "Python-like source types mapped to JSON Schema types, including nested list, tuple, and union contents",
        },
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_method": "Shuffle whitespace-normalized query groups; fill ceil(test_fraction * rows) test rows without splitting groups; shuffle each output",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "sha256": output_hashes,
        "prepare_script_sha256": file_sha256(__file__),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("huggingface-hub",)
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"Loaded {len(source_rows):,} rows; removed {invalid_rows:,} invalid rows."
    )
    if duplicate_tool_name_rows:
        print(
            f"  {duplicate_tool_name_rows:,} rows had ambiguous duplicate "
            "available-tool names."
        )
    print(
        f"Saved {len(train_rows):,} train / {len(test_rows):,} test rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
