# Minimal GRPO in PyTorch

This tutorial is a self-contained GRPO implementation for NVIDIA GPUs. It
places GSM8K preparation, answer rewards, rollout generation, the complete
GRPO training calculation, validation, and checkpoint saving in one readable
Python file.

The implementation is assembled from the proven code in
[`cuda_backend/grpo_train.py`](../cuda_backend/grpo_train.py),
[`cuda_backend/gsm8k_eval.py`](../cuda_backend/gsm8k_eval.py), and
[`data_preparation/gsm8k.py`](../data_preparation/gsm8k.py). The reused code is
kept unchanged apart from replacing command-line arguments with constants and
replacing cross-module references with same-file references.

## Requirements

- Python 3.12 or newer
- An NVIDIA GPU with CUDA support
- [`uv`](https://github.com/astral-sh/uv)

Install the CUDA dependencies from the repository root:

```bash
uv sync --extra cuda
```

## Run the tutorial

Run the script from the repository root:

```bash
uv run --extra cuda python tutorials/grpo_minimal_pytorch.py
```

The script has no command-line arguments. Model, data, GRPO, evaluation, and
checkpoint settings are uppercase constants at the beginning of
[`grpo_minimal_pytorch.py`](grpo_minimal_pytorch.py).

## Evaluation and terminal output

The script evaluates greedy GSM8K answer accuracy on the official test set
before training and after training so that the final result can be compared to
the starting model. The test result is never used for optimization, checkpoint
selection, or early stopping.

During training, a held-out portion of the GSM8K training split is evaluated
at step 0 and every 50 steps. The terminal shows the current training loss,
validation and test accuracies, and saved checkpoint paths. GRPO does not have
a supervised validation loss here; answer accuracy is the validation metric.

## Checkpoints

The LoRA adapter and tokenizer are saved every 100 steps and at the final step
when needed:

```text
checkpoints/tutorials/grpo_minimal_pytorch/step_XXXXXX/
```

The root `checkpoints/` directory is ignored by Git, so generated weights are
not committed with the tutorial source.

## Reproducibility

The script uses the same deterministic CUDA seeding setup as the full CUDA
implementation. Reproducibility still depends on using compatible hardware,
CUDA, PyTorch, and dependency versions.

For configurable training, adapter continuation, full fine-tuning, TensorBoard
logging, timestamped runs, and additional experiment management, use the
[full CUDA GRPO implementation](../cuda_backend/grpo_train.py).
