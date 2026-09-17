"""Evaluate a base model with the paper's fixed benchmark protocol."""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts import evaluate_checkpoints as checkpoint_evaluator


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "dump"
REPORT_NAME = "base_model_evaluation.md"
FULL_REPORT_NAME = "base_model_evaluation_full.md"
CSV_NAME = "base_model_evaluation.csv"
SPREADSHEET_HEADERS = (
    "MMLU",
    "MATH-500",
    "IFEval",
    "xLAM ETC",
    "xLAM FNA",
    "BFCL Single-turn FC",
    "Relevance",
    "Irrelevance",
)
BFCL_FUNCTION_CALLING_SUBSETS = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
EXPECTED_BFCL_SAMPLE_COUNTS = {
    "simple": 400,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "live_simple": 258,
    "live_multiple": 1053,
    "live_parallel": 16,
    "live_parallel_multiple": 24,
    "irrelevance": 240,
    "live_relevance": 18,
    "live_irrelevance": 882,
}


def parse_args(argv=None):
    """Parse the model, output location, and overwrite policy."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one base model on xLAM, selected BFCL-v3, MMLU, "
            "MATH-500, and IFEval, then write spreadsheet-ready results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local full-model path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the concise, full, and CSV result files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace result files that already exist",
    )
    return parser.parse_args(argv)


def run_xlam(model, temporary_path):
    """Run the local xLAM holdout directly on a full base model."""
    predictions_path = temporary_path / "xlam_predictions.jsonl"
    log_path = temporary_path / "xlam.log"
    command = [
        "uv",
        "run",
        "--extra",
        "cuda",
        "-m",
        "cuda_backend.xlam_function_calling_eval",
        "--model_path",
        model,
        "--output-jsonl",
        str(predictions_path),
        "--max-prompt-len",
        "640",
        "--max-new-tokens",
        str(checkpoint_evaluator.XLAM_MAX_NEW_TOKENS),
        "--batch-size",
        "8",
    ]
    checkpoint_evaluator.run_logged(command, log_path)
    return checkpoint_evaluator.parse_xlam_results(predictions_path, log_path)


def paper_bfcl_metrics(metrics):
    """Calculate the paper's three BFCL metrics from selected subset scores."""
    for name, expected_count in EXPECTED_BFCL_SAMPLE_COUNTS.items():
        actual_count = metrics[name]["samples"]
        if actual_count != expected_count:
            raise ValueError(
                f"BFCL {name} returned {actual_count} samples; "
                f"the recorded protocol requires {expected_count}"
            )

    function_calling_count = sum(
        metrics[name]["samples"] for name in BFCL_FUNCTION_CALLING_SUBSETS
    )
    function_calling_correct = sum(
        metrics[name]["samples"] * metrics[name]["accuracy"]
        for name in BFCL_FUNCTION_CALLING_SUBSETS
    )
    irrelevance_subsets = ("irrelevance", "live_irrelevance")
    irrelevance_count = sum(
        metrics[name]["samples"] for name in irrelevance_subsets
    )
    irrelevance_correct = sum(
        metrics[name]["samples"] * metrics[name]["accuracy"]
        for name in irrelevance_subsets
    )
    return {
        "single_turn_fc": function_calling_correct / function_calling_count,
        "relevance": metrics["live_relevance"]["accuracy"],
        "irrelevance": irrelevance_correct / irrelevance_count,
    }


def run_bfcl(model, temporary_path):
    """Serve a full base model and run the selected BFCL-v3 subsets."""
    port = checkpoint_evaluator.find_free_port()
    server_log_path = temporary_path / "vllm_server.log"
    eval_log_path = temporary_path / "bfcl.log"
    work_dir = temporary_path / "bfcl"
    environment = os.environ.copy()
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    server_command = [
        "uv",
        "run",
        "--extra",
        "evaluate",
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    dataset_args = {
        "bfcl_v3": {
            "subset_list": list(checkpoint_evaluator.BFCL_SUBSETS),
            "extra_params": {
                "is_fc_model": True,
                "underscore_to_dot": True,
            },
        }
    }
    generation_config = {
        "temperature": 0,
        "max_tokens": 4096,
        "parallel_tool_calls": True,
    }
    eval_command = [
        "uv",
        "run",
        "--extra",
        "evaluate",
        "evalscope",
        "eval",
        "--model",
        model,
        "--api-url",
        f"http://127.0.0.1:{port}/v1",
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        "bfcl_v3",
        "--dataset-args",
        json.dumps(dataset_args, separators=(",", ":")),
        "--generation-config",
        json.dumps(generation_config, separators=(",", ":")),
        "--eval-batch-size",
        "8",
        "--work-dir",
        str(work_dir),
        "--no-timestamp",
    ]

    process = None
    with server_log_path.open("w", encoding="utf-8") as server_log_file:
        try:
            process = subprocess.Popen(
                server_command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=server_log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            checkpoint_evaluator.wait_for_vllm(process, port, server_log_path)
            checkpoint_evaluator.run_logged(
                eval_command,
                eval_log_path,
                environment=environment,
            )
        finally:
            checkpoint_evaluator.stop_vllm_process(process)

    metrics = checkpoint_evaluator.parse_bfcl_results(work_dir)
    metrics["_paper_metrics"] = paper_bfcl_metrics(metrics)
    return metrics


def run_lm_eval(model, temporary_path):
    """Run MMLU, MATH-500, and IFEval directly on a full base model."""
    output_dir = temporary_path / "lm_eval"
    log_path = temporary_path / "lm_eval.log"
    environment = os.environ.copy()
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    model_args = ",".join(
        (
            f"pretrained={model}",
            "dtype=bfloat16",
            "gpu_memory_utilization=0.85",
        )
    )
    command = [
        "uv",
        "run",
        "--extra",
        "evaluate",
        "lm-eval",
        "run",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        "mmlu,minerva_math500,ifeval",
        "--apply_chat_template",
        "--batch_size",
        "auto",
        "--output_path",
        str(output_dir),
    ]
    checkpoint_evaluator.run_logged(command, log_path, environment=environment)
    return checkpoint_evaluator.parse_lm_eval_results(output_dir)


def evaluate_suite(title, function):
    """Run one suite while keeping the terminal informed of coarse progress."""
    print(f"Evaluating {title} ...", flush=True)
    result = checkpoint_evaluator.capture_suite(function)
    print(f"  {result['status']}", flush=True)
    return result


def evaluate_base_model(model, temporary_path):
    """Attempt every fixed evaluation suite for one base model."""
    return {
        "model": model,
        "suites": {
            "xlam": evaluate_suite(
                "xLAM",
                lambda: run_xlam(model, temporary_path),
            ),
            "bfcl": evaluate_suite(
                "BFCL-v3",
                lambda: run_bfcl(model, temporary_path),
            ),
            "lm_eval": evaluate_suite(
                "MMLU, MATH-500, and IFEval",
                lambda: run_lm_eval(model, temporary_path),
            ),
        },
    }


def spreadsheet_values(evaluation):
    """Return the eight requested spreadsheet values in header order."""
    values = {header: None for header in SPREADSHEET_HEADERS}
    xlam = evaluation["suites"]["xlam"]
    if xlam["status"] == "success":
        values["xLAM ETC"] = xlam["metrics"]["exact_accuracy"]
        values["xLAM FNA"] = xlam["metrics"]["name_accuracy"]

    bfcl = evaluation["suites"]["bfcl"]
    if bfcl["status"] == "success":
        paper_metrics = bfcl["metrics"]["_paper_metrics"]
        values["BFCL Single-turn FC"] = paper_metrics["single_turn_fc"]
        values["Relevance"] = paper_metrics["relevance"]
        values["Irrelevance"] = paper_metrics["irrelevance"]

    lm_eval = evaluation["suites"]["lm_eval"]
    if lm_eval["status"] == "success":
        values["MMLU"] = lm_eval["metrics"]["mmlu_accuracy"]
        values["MATH-500"] = lm_eval["metrics"]["math_verify"]
        values["IFEval"] = lm_eval["metrics"]["ifeval_prompt_strict"]
    return values


def render_markdown(evaluation):
    """Render the status and one spreadsheet-ready result row."""
    suite_titles = {
        "xlam": "xLAM 640-token-filtered local holdout",
        "bfcl": "BFCL-v3 selected single-turn suite",
        "lm_eval": "MMLU, MATH-500, and IFEval",
    }
    values = spreadsheet_values(evaluation)
    lines = [
        "# Base Model Evaluation",
        "",
        f"- Model: `{evaluation['model']}`",
        "",
        "## Status",
        "",
        "| Suite | Status |",
        "| --- | --- |",
    ]
    for name, result in evaluation["suites"].items():
        lines.append(f"| {suite_titles[name]} | {result['status'].title()} |")

    lines.extend(
        [
            "",
            "## Spreadsheet values",
            "",
            "| " + " | ".join(SPREADSHEET_HEADERS) + " |",
            "| " + " | ".join("---:" for _ in SPREADSHEET_HEADERS) + " |",
            "| "
            + " | ".join(
                checkpoint_evaluator.percentage(values[header])
                if values[header] is not None
                else "—"
                for header in SPREADSHEET_HEADERS
            )
            + " |",
        ]
    )

    failures = [
        (suite_titles[name], result)
        for name, result in evaluation["suites"].items()
        if result["status"] == "failed"
    ]
    if failures:
        lines.extend(["", "## Failures"])
        for title, failure in failures:
            exit_code = failure.get("exit_code")
            exit_text = f" Exit code: `{exit_code}`." if exit_code is not None else ""
            lines.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    f"{failure['error']}{exit_text}",
                    "",
                    "```text",
                    checkpoint_evaluator.markdown_diagnostic(
                        failure.get("diagnostic", "")
                    ),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def render_full_markdown(evaluation):
    """Render the concise result plus complete evaluator summaries."""
    compact_report = render_markdown(evaluation).replace(
        "# Base Model Evaluation",
        "# Full Base Model Evaluation",
        1,
    )
    lines = [
        compact_report.rstrip(),
        "",
        "## Complete evaluator data",
        "",
        "This retains summary results and metadata, but not predictions or logs.",
    ]

    xlam = evaluation["suites"]["xlam"]
    if xlam["status"] == "success":
        metrics = xlam["metrics"]
        lines.extend(
            [
                "",
                "### xLAM complete summary",
                "",
                (
                    "| Evaluated | Skipped | Parsed completions | No EOS | "
                    "Exact accuracy | Name accuracy |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| {metrics['evaluated']} | {metrics['skipped']} | "
                    f"{metrics['parsed']} | {metrics['truncated']} | "
                    f"{metrics['exact_accuracy']} | {metrics['name_accuracy']} |"
                ),
            ]
        )

    bfcl = evaluation["suites"]["bfcl"]
    if bfcl["status"] == "success":
        metrics = bfcl["metrics"]
        paper_metrics = metrics["_paper_metrics"]
        lines.extend(
            [
                "",
                "### BFCL paper metrics",
                "",
                "| Metric | Accuracy |",
                "| --- | ---: |",
                f"| Single-turn FC | {paper_metrics['single_turn_fc']} |",
                f"| Relevance | {paper_metrics['relevance']} |",
                f"| Irrelevance | {paper_metrics['irrelevance']} |",
                "",
                "### BFCL selected subsets",
                "",
                "| Subset | Samples | Accuracy |",
                "| --- | ---: | ---: |",
            ]
        )
        for name in checkpoint_evaluator.BFCL_SUBSETS:
            result = metrics[name]
            lines.append(
                f"| {name} | {result['samples']} | {result['accuracy']} |"
            )
        full_report = metrics.get("_full_report")
        if full_report is not None:
            lines.extend(
                [
                    "",
                    *checkpoint_evaluator.render_json_details(
                        "Complete BFCL result JSON",
                        full_report,
                    ),
                ]
            )

    lm_eval = evaluation["suites"]["lm_eval"]
    if lm_eval["status"] == "success":
        full_results = lm_eval["metrics"].get("_full_results")
        if full_results is not None:
            if full_results.get("groups"):
                lines.extend(
                    [
                        "",
                        *checkpoint_evaluator.render_lm_eval_metric_table(
                            full_results,
                            "groups",
                            "lm-eval groups",
                        ),
                    ]
                )
            lines.extend(
                [
                    "",
                    *checkpoint_evaluator.render_lm_eval_metric_table(
                        full_results,
                        "results",
                        "lm-eval tasks",
                    ),
                    "",
                    *checkpoint_evaluator.render_json_details(
                        "Complete lm-eval result JSON",
                        full_results,
                    ),
                ]
            )
    return "\n".join(lines) + "\n"


def render_csv(evaluation):
    """Render the requested headers and scores as an Excel-compatible CSV."""
    values = spreadsheet_values(evaluation)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(SPREADSHEET_HEADERS)
    writer.writerow(
        checkpoint_evaluator.percentage(values[header])
        if values[header] is not None
        else ""
        for header in SPREADSHEET_HEADERS
    )
    return output.getvalue()


def output_paths(output_dir):
    """Return all generated artifact paths for one output directory."""
    return (
        output_dir / REPORT_NAME,
        output_dir / FULL_REPORT_NAME,
        output_dir / CSV_NAME,
    )


def ensure_outputs_available(paths, force):
    """Reject accidental replacement unless the caller requested it."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise ValueError(
            f"Output files already exist: {names}; pass --force to replace"
        )


def write_outputs(output_dir, evaluation):
    """Atomically write concise, full, and spreadsheet-ready reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path, full_report_path, csv_path = output_paths(output_dir)
    checkpoint_evaluator.write_report_atomic(
        report_path,
        render_markdown(evaluation),
    )
    checkpoint_evaluator.write_report_atomic(
        full_report_path,
        render_full_markdown(evaluation),
    )
    checkpoint_evaluator.write_report_atomic(csv_path, render_csv(evaluation))
    return report_path, full_report_path, csv_path


def main(argv=None):
    """Run and record the fixed base-model evaluation."""
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        paths = output_paths(output_dir)
        ensure_outputs_available(paths, args.force)
        with tempfile.TemporaryDirectory(prefix="base-model-evaluation-") as directory:
            evaluation = evaluate_base_model(args.model, Path(directory))
        written_paths = write_outputs(output_dir, evaluation)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for path in written_paths:
        print(f"Wrote {path}")
    failures = checkpoint_evaluator.collect_failures(
        {
            "checkpoint": args.model,
            "suites": evaluation["suites"],
        }
    )
    checkpoint_evaluator.print_failures(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
