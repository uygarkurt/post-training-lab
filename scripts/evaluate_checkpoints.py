"""Evaluate every LoRA checkpoint in one training run with fixed benchmarks."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_NAME = "evaluation_results.md"
FULL_REPORT_NAME = "evaluation_results_full.md"
STEP_PATTERN = re.compile(r"step_(\d+)")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BFCL_SUBSETS = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_relevance",
    "live_irrelevance",
)
SERVER_STARTUP_TIMEOUT_SECONDS = 600
SERVER_SHUTDOWN_TIMEOUT_SECONDS = 30
DIAGNOSTIC_LINE_LIMIT = 20
DIAGNOSTIC_CHARACTER_LIMIT = 4000
XLAM_MAX_NEW_TOKENS = 256


class EvaluationFailure(RuntimeError):
    """Carry a concise evaluator failure into the Markdown report."""

    def __init__(self, message, diagnostic="", exit_code=None):
        """Store the reportable message, log tail, and optional exit code."""
        super().__init__(message)
        self.diagnostic = diagnostic
        self.exit_code = exit_code


def parse_args(argv=None):
    """Parse the checkpoint-run path and rerun policy."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every step_<number> LoRA adapter in a training run and "
            f"write {REPORT_NAME} and {FULL_REPORT_NAME} inside each checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoints_path",
        type=Path,
        help="Directory whose immediate step_<number> children are LoRA checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reevaluate checkpoints that already contain both reports",
    )
    return parser.parse_args(argv)


def discover_checkpoints(checkpoints_path):
    """Return immediate step directories sorted by their numeric suffix."""
    checkpoints_path = checkpoints_path.expanduser().resolve()
    if not checkpoints_path.is_dir():
        raise ValueError(f"Checkpoint run directory not found: {checkpoints_path}")

    checkpoints = []
    for child in checkpoints_path.iterdir():
        match = STEP_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise ValueError(
            f"No immediate step_<number> directories found in {checkpoints_path}"
        )
    return [checkpoint for _, checkpoint in checkpoints]


def load_adapter_metadata(checkpoint_path):
    """Validate a PEFT LoRA checkpoint and return its base model and maximum rank."""
    config_path = checkpoint_path / "adapter_config.json"
    weight_paths = (
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "adapter_model.bin",
    )
    if not config_path.is_file():
        raise ValueError(f"Missing adapter_config.json in {checkpoint_path}")
    if not any(path.is_file() for path in weight_paths):
        raise ValueError(f"Missing adapter weights in {checkpoint_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {config_path}: {error}") from error

    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Checkpoint is not a LoRA adapter: {checkpoint_path}")
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(f"Missing base_model_name_or_path in {config_path}")

    ranks = [config.get("r")]
    rank_pattern = config.get("rank_pattern") or {}
    if not isinstance(rank_pattern, dict):
        raise ValueError(f"rank_pattern must be an object in {config_path}")
    ranks.extend(rank_pattern.values())
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 for rank in ranks):
        raise ValueError(f"LoRA ranks must be positive integers in {config_path}")

    return {"base_model": base_model, "max_lora_rank": max(ranks)}


def display_path(path):
    """Format a checkpoint path relative to the repository when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return os.path.relpath(path.resolve(), Path.cwd())


def read_log_tail(log_path):
    """Return a short, ANSI-free diagnostic from the end of a captured log."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"Could not read captured log: {error}"
    text = ANSI_ESCAPE_PATTERN.sub("", text).replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    diagnostic = "\n".join(lines[-DIAGNOSTIC_LINE_LIMIT:])
    if len(diagnostic) > DIAGNOSTIC_CHARACTER_LIMIT:
        diagnostic = diagnostic[-DIAGNOSTIC_CHARACTER_LIMIT:]
    return diagnostic or "No diagnostic output was captured."


def run_logged(command, log_path, environment=None):
    """Run one command with all output redirected to a temporary log."""
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise EvaluationFailure(
            f"Process exited with status {completed.returncode}.",
            diagnostic=read_log_tail(log_path),
            exit_code=completed.returncode,
        )


def parse_xlam_results(predictions_path, log_path):
    """Calculate headline xLAM metrics from its temporary JSONL predictions."""
    evaluated = 0
    exact_correct = 0
    name_correct = 0
    parsed = 0
    truncated = 0
    try:
        with predictions_path.open(encoding="utf-8") as predictions_file:
            for line_number, line in enumerate(predictions_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid xLAM JSONL at line {line_number}: {error}"
                    ) from error
                evaluated += 1
                exact_correct += bool(record.get("correct"))
                name_correct += bool(record.get("names_correct"))
                parsed += bool(record.get("predicted_calls"))
                truncated += bool(record.get("truncated"))
    except OSError as error:
        raise ValueError(f"Could not read xLAM predictions: {error}") from error
    if evaluated == 0:
        raise ValueError("xLAM produced no prediction records")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) samples loaded,\s*(\d+) skipped", log_text)
    if not match:
        raise ValueError("Could not recover xLAM loaded/skipped counts from its log")
    loaded = int(match.group(1))
    skipped = int(match.group(2))
    if loaded != evaluated:
        raise ValueError(
            f"xLAM reported {loaded} loaded samples but wrote {evaluated} predictions"
        )
    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "parsed": parsed,
        "truncated": truncated,
        "exact_accuracy": exact_correct / evaluated,
        "name_accuracy": name_correct / evaluated,
    }


def find_single_file(root_path, pattern, description):
    """Find the one result file produced in a fresh temporary directory."""
    matches = sorted(root_path.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {description} under {root_path}, found {len(matches)}"
        )
    return matches[0]


def numeric_value(value, description):
    """Validate and convert one persisted metric to float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or non-numeric metric: {description}")
    return float(value)


def parse_bfcl_results(work_dir):
    """Extract selected BFCL subsets and aggregates from an EvalScope report."""
    report_path = find_single_file(
        work_dir / "reports", "bfcl_v3.json", "BFCL report"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    subsets = {}
    for metric in report.get("metrics", []):
        for category in metric.get("categories", []):
            for subset in category.get("subsets", []):
                name = subset.get("name")
                if isinstance(name, str):
                    subsets[name.casefold()] = subset

    required = (
        "non_live",
        "live",
        "overall",
        *BFCL_SUBSETS,
    )
    results = {}
    for name in required:
        if name not in subsets:
            raise ValueError(f"BFCL report is missing the {name} subset")
        subset = subsets[name]
        count = subset.get("num")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"BFCL report has an invalid sample count for {name}")
        results[name] = {
            "accuracy": numeric_value(subset.get("score"), f"BFCL {name}"),
            "samples": count,
        }
    results["_full_report"] = report
    return results


def metric_from_task(results, section, task, metric):
    """Read one named lm-eval metric from a task or group result."""
    task_metrics = results.get(section, {}).get(task)
    if not isinstance(task_metrics, dict):
        raise ValueError(f"lm-eval results are missing {section}.{task}")
    return numeric_value(task_metrics.get(metric), f"lm-eval {task} {metric}")


def parse_lm_eval_results(output_dir):
    """Extract headline metrics and retain the complete lm-eval result."""
    result_path = find_single_file(
        output_dir, "results_*.json", "lm-eval result file"
    )
    results = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "mmlu_accuracy": metric_from_task(results, "groups", "mmlu", "acc,none"),
        "math_exact_match": metric_from_task(
            results, "results", "minerva_math500", "exact_match,none"
        ),
        "math_verify": metric_from_task(
            results, "results", "minerva_math500", "math_verify,none"
        ),
        "ifeval_prompt_strict": metric_from_task(
            results, "results", "ifeval", "prompt_level_strict_acc,none"
        ),
        "ifeval_instruction_strict": metric_from_task(
            results, "results", "ifeval", "inst_level_strict_acc,none"
        ),
        "_full_results": results,
    }


def run_xlam(checkpoint_path, temporary_path):
    """Run the local xLAM test evaluator for one adapter."""
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
        str(checkpoint_path),
        "--output-jsonl",
        str(predictions_path),
        "--load-adapter",
        "--max-prompt-len",
        "640",
        "--max-new-tokens",
        str(XLAM_MAX_NEW_TOKENS),
        "--batch-size",
        "8",
    ]
    run_logged(command, log_path)
    return parse_xlam_results(predictions_path, log_path)


def find_free_port():
    """Ask the operating system for an unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_vllm(process, port, log_path):
    """Poll the vLLM model endpoint until it accepts requests or times out."""
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise EvaluationFailure(
                f"vLLM server exited with status {return_code} before becoming ready.",
                diagnostic=read_log_tail(log_path),
                exit_code=return_code,
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise EvaluationFailure(
        f"vLLM server did not become ready within {SERVER_STARTUP_TIMEOUT_SECONDS} seconds.",
        diagnostic=read_log_tail(log_path),
    )


def stop_vllm_process(process):
    """Terminate the complete process group owned by a vLLM server."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def run_bfcl(checkpoint_path, adapter_metadata, temporary_path):
    """Serve one adapter with vLLM and evaluate the selected BFCL subsets."""
    port = find_free_port()
    model_name = checkpoint_path.name
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
        adapter_metadata["base_model"],
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enable-lora",
        "--lora-modules",
        f"{model_name}={checkpoint_path}",
        "--max-lora-rank",
        str(adapter_metadata["max_lora_rank"]),
    ]
    dataset_args = {
        "bfcl_v3": {
            "subset_list": list(BFCL_SUBSETS),
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
        model_name,
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
            wait_for_vllm(process, port, server_log_path)
            run_logged(eval_command, eval_log_path, environment=environment)
        finally:
            stop_vllm_process(process)
    return parse_bfcl_results(work_dir)


def run_lm_eval(checkpoint_path, adapter_metadata, temporary_path):
    """Run MMLU, MATH-500, and IFEval through lm-eval's vLLM backend."""
    output_dir = temporary_path / "lm_eval"
    log_path = temporary_path / "lm_eval.log"
    environment = os.environ.copy()
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    model_args = ",".join(
        (
            f"pretrained={adapter_metadata['base_model']}",
            f"lora_local_path={checkpoint_path}",
            "dtype=bfloat16",
            f"max_lora_rank={adapter_metadata['max_lora_rank']}",
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
    run_logged(command, log_path, environment=environment)
    return parse_lm_eval_results(output_dir)


def capture_suite(function):
    """Convert one evaluator call into a success or failure result."""
    try:
        return {"status": "success", "metrics": function()}
    except EvaluationFailure as error:
        return {
            "status": "failed",
            "error": str(error),
            "exit_code": error.exit_code,
            "diagnostic": error.diagnostic,
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "exit_code": None,
            "diagnostic": "",
        }


def evaluate_checkpoint(checkpoint_path, temporary_path):
    """Attempt every requested suite for one checkpoint."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        adapter_metadata = load_adapter_metadata(checkpoint_path)
    except Exception as error:
        return {
            "checkpoint": display_path(checkpoint_path),
            "timestamp": timestamp,
            "base_model": None,
            "max_lora_rank": None,
            "suites": {},
            "validation": {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "exit_code": None,
                "diagnostic": "",
            },
        }

    suites = {
        "xlam": capture_suite(
            lambda: run_xlam(checkpoint_path, temporary_path)
        ),
        "bfcl": capture_suite(
            lambda: run_bfcl(checkpoint_path, adapter_metadata, temporary_path)
        ),
        "lm_eval": capture_suite(
            lambda: run_lm_eval(checkpoint_path, adapter_metadata, temporary_path)
        ),
    }
    return {
        "checkpoint": display_path(checkpoint_path),
        "timestamp": timestamp,
        "base_model": adapter_metadata["base_model"],
        "max_lora_rank": adapter_metadata["max_lora_rank"],
        "suites": suites,
    }


def percentage(value):
    """Format a fractional accuracy as a percentage."""
    return f"{value * 100:.2f}%"


def markdown_diagnostic(text):
    """Make a captured diagnostic safe for a Markdown code fence."""
    return (text or "No diagnostic output was captured.").replace("```", "` ` `")


def render_markdown(evaluation):
    """Render a concise human-readable checkpoint evaluation report."""
    suite_titles = {
        "xlam": "xLAM test set",
        "bfcl": "BFCL-v3 selected single-turn suite",
        "lm_eval": "General benchmarks",
    }
    rank = evaluation["max_lora_rank"]
    rank_text = rank if rank is not None else "unknown"
    lines = [
        "# Evaluation Results",
        "",
        f"- Checkpoint: `{evaluation['checkpoint']}`",
        f"- Base model: `{evaluation['base_model'] or 'unknown'}`",
        f"- Maximum LoRA rank: `{rank_text}`",
        f"- Evaluated at: `{evaluation['timestamp']}`",
        "",
        "## Status",
        "",
        "| Suite | Status |",
        "| --- | --- |",
    ]
    if "validation" in evaluation:
        lines.append("| Checkpoint validation | Failed |")
    for name, result in evaluation["suites"].items():
        lines.append(f"| {suite_titles[name]} | {result['status'].title()} |")

    xlam = evaluation["suites"].get("xlam")
    if xlam and xlam["status"] == "success":
        metrics = xlam["metrics"]
        lines.extend(
            [
                "",
                "## xLAM test set",
                "",
                (
                    "| Evaluated | Skipped | No EOS "
                    f"(hit {XLAM_MAX_NEW_TOKENS}-token limit) | "
                    "Exact tool-call accuracy | Function-name accuracy |"
                ),
                "| ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| {metrics['evaluated']} | {metrics['skipped']} | "
                    f"{metrics['truncated']} | "
                    f"{percentage(metrics['exact_accuracy'])} | "
                    f"{percentage(metrics['name_accuracy'])} |"
                ),
            ]
        )

    bfcl = evaluation["suites"].get("bfcl")
    if bfcl and bfcl["status"] == "success":
        metrics = bfcl["metrics"]
        aggregate_labels = (
            ("non_live", "Selected-suite non-live"),
            ("live", "Selected-suite live"),
            ("overall", "Selected-suite overall (not full BFCL-v3)"),
        )
        lines.extend(
            [
                "",
                "## BFCL-v3 selected single-turn suite",
                "",
                "### Aggregates",
                "",
                "| Metric | Samples | Accuracy |",
                "| --- | ---: | ---: |",
            ]
        )
        for name, label in aggregate_labels:
            result = metrics[name]
            lines.append(
                f"| {label} | {result['samples']} | {percentage(result['accuracy'])} |"
            )
        lines.extend(
            [
                "",
                "### Selected subsets",
                "",
                "| Subset | Samples | Accuracy |",
                "| --- | ---: | ---: |",
            ]
        )
        for name in BFCL_SUBSETS:
            result = metrics[name]
            label = name.replace("_", " ").title()
            lines.append(
                f"| {label} | {result['samples']} | {percentage(result['accuracy'])} |"
            )

    lm_eval = evaluation["suites"].get("lm_eval")
    if lm_eval and lm_eval["status"] == "success":
        metrics = lm_eval["metrics"]
        rows = (
            ("MMLU accuracy", "mmlu_accuracy"),
            ("MATH-500 math verify", "math_verify"),
            ("IFEval prompt-level strict accuracy", "ifeval_prompt_strict"),
        )
        lines.extend(
            [
                "",
                "## General benchmarks",
                "",
                "| Metric | Score |",
                "| --- | ---: |",
            ]
        )
        for label, name in rows:
            lines.append(f"| {label} | {percentage(metrics[name])} |")

    failures = []
    if "validation" in evaluation:
        failures.append(("Checkpoint validation", evaluation["validation"]))
    failures.extend(
        (suite_titles[name], result)
        for name, result in evaluation["suites"].items()
        if result["status"] == "failed"
    )
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
                    markdown_diagnostic(failure.get("diagnostic", "")),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def markdown_cell(value):
    """Escape a scalar value for use in a Markdown table cell."""
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_lm_eval_metric_table(results, section, heading):
    """Render every scalar lm-eval metric in one result section."""
    lines = [
        f"### {heading}",
        "",
        "| Name | Version | Filter | n-shot | Metric | Direction | Samples | Value | Stderr |",
        "| --- | --- | --- | ---: | --- | :---: | ---: | ---: | ---: |",
    ]
    result_rows = results.get(section, {})
    versions = results.get("versions", {})
    n_shot = results.get("n-shot", {})
    higher_is_better = results.get("higher_is_better", {})
    row_count = 0
    for name, metric_values in result_rows.items():
        if not isinstance(metric_values, dict):
            continue
        sample_counts = metric_values.get("sample_count", {})
        if not isinstance(sample_counts, dict):
            sample_counts = {}
        sample_length = metric_values.get("sample_len", "")
        for metric_key, value in metric_values.items():
            if metric_key in {"name", "alias", "sample_len", "sample_count"}:
                continue
            if not isinstance(value, (str, int, float, bool)):
                continue
            metric_name, separator, filter_name = metric_key.rpartition(",")
            if not separator:
                metric_name = metric_key
                filter_name = ""
            if metric_name.endswith("_stderr"):
                continue
            stderr_name = f"{metric_name}_stderr"
            stderr_key = (
                f"{stderr_name},{filter_name}" if filter_name else stderr_name
            )
            stderr = metric_values.get(stderr_key, "")
            direction_value = higher_is_better.get(name, {}).get(metric_name)
            if direction_value is True:
                direction = "↑"
            elif direction_value is False:
                direction = "↓"
            else:
                direction = ""
            samples = sample_counts.get(metric_key, sample_length)
            lines.append(
                "| "
                f"{markdown_cell(metric_values.get('alias', name))} | "
                f"{markdown_cell(versions.get(name))} | "
                f"{markdown_cell(filter_name)} | "
                f"{markdown_cell(n_shot.get(name))} | "
                f"{markdown_cell(metric_name)} | {markdown_cell(direction)} | "
                f"{markdown_cell(samples)} | {markdown_cell(value)} | "
                f"{markdown_cell(stderr)} |"
            )
            row_count += 1
    if row_count == 0:
        lines.extend(["", "No metrics were returned for this section."])
    return lines


def render_json_details(title, result):
    """Render complete evaluator result data in a collapsible JSON block."""
    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    serialized = serialized.replace("```", "` ` `")
    return [
        "<details>",
        f"<summary>{title}</summary>",
        "",
        "```json",
        serialized,
        "```",
        "",
        "</details>",
    ]


def render_full_markdown(evaluation):
    """Render complete evaluator tables and result payloads for one checkpoint."""
    compact_report = render_markdown(evaluation)
    compact_report = compact_report.replace(
        "# Evaluation Results",
        "# Full Evaluation Results",
        1,
    )
    lines = [
        compact_report.rstrip(),
        "",
        "## Complete evaluator data",
        "",
        "This retains summary results and metadata, but not predictions or logs.",
    ]

    xlam = evaluation["suites"].get("xlam")
    if xlam and xlam["status"] == "success":
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

    bfcl = evaluation["suites"].get("bfcl")
    if bfcl and bfcl["status"] == "success":
        full_report = bfcl["metrics"].get("_full_report")
        if full_report is not None:
            lines.extend(
                [
                    "",
                    *render_json_details(
                        "Complete BFCL result JSON",
                        full_report,
                    ),
                ]
            )

    lm_eval = evaluation["suites"].get("lm_eval")
    if lm_eval and lm_eval["status"] == "success":
        full_results = lm_eval["metrics"].get("_full_results")
        if full_results is not None:
            if full_results.get("groups"):
                lines.extend(
                    [
                        "",
                        *render_lm_eval_metric_table(
                            full_results,
                            "groups",
                            "lm-eval groups",
                        ),
                    ]
                )
            lines.extend(
                [
                    "",
                    *render_lm_eval_metric_table(
                        full_results,
                        "results",
                        "lm-eval tasks",
                    ),
                    "",
                    *render_json_details(
                        "Complete lm-eval result JSON",
                        full_results,
                    ),
                ]
            )
    return "\n".join(lines) + "\n"


def write_report_atomic(report_path, content):
    """Replace a checkpoint report only after its new content is complete."""
    temporary_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            output_file.write(content)
            temporary_file = Path(output_file.name)
        os.replace(temporary_file, report_path)
    finally:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()


def collect_failures(evaluation):
    """Return concise failure records for the final terminal summary."""
    failures = []
    if "validation" in evaluation:
        failures.append((evaluation["checkpoint"], "validation", evaluation["validation"]))
    failures.extend(
        (evaluation["checkpoint"], name, result)
        for name, result in evaluation["suites"].items()
        if result["status"] == "failed"
    )
    return failures


def evaluate_all(checkpoints_path, force=False):
    """Evaluate or skip every checkpoint and return all recorded failures."""
    checkpoints = discover_checkpoints(checkpoints_path)
    failures = []
    with tqdm(
        total=len(checkpoints),
        desc="Checkpoints",
        unit="checkpoint",
        dynamic_ncols=True,
    ) as progress:
        for checkpoint_path in checkpoints:
            report_path = checkpoint_path / REPORT_NAME
            full_report_path = checkpoint_path / FULL_REPORT_NAME
            reports_exist = report_path.exists() and full_report_path.exists()
            if reports_exist and not force:
                progress.update(1)
                continue

            with tempfile.TemporaryDirectory(prefix="checkpoint-evaluation-") as directory:
                evaluation = evaluate_checkpoint(checkpoint_path, Path(directory))
            try:
                compact_report = render_markdown(evaluation)
                full_report = render_full_markdown(evaluation)
                write_report_atomic(full_report_path, full_report)
                write_report_atomic(report_path, compact_report)
            except Exception as error:
                failures.append(
                    (
                        display_path(checkpoint_path),
                        "reports",
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "exit_code": None,
                        },
                    )
                )
            else:
                failures.extend(collect_failures(evaluation))
            progress.update(1)
    return failures


def print_failures(failures):
    """Print the permitted end-of-run failure summary."""
    if not failures:
        return
    print("Failures:", file=sys.stderr)
    for checkpoint, suite, failure in failures:
        exit_code = failure.get("exit_code")
        exit_text = f" (exit {exit_code})" if exit_code is not None else ""
        print(
            f"  {checkpoint}: {suite}{exit_text}: {failure['error']}",
            file=sys.stderr,
        )


def main(argv=None):
    """Run the checkpoint evaluation batch."""
    args = parse_args(argv)
    try:
        failures = evaluate_all(args.checkpoints_path, force=args.force)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print_failures(failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
