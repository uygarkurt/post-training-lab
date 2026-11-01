# CUDA Backend

The CUDA backend is the runnable PyTorch implementation of the post-training
pipeline for NVIDIA GPUs. SFT, GRPO, and OPD keep their training math visible and
support LoRA so experiments can run on consumer hardware.

Run all commands below from the repository root.

## Requirements

- NVIDIA GPU with CUDA support
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

Install the shared project dependencies and the CUDA extra:

```bash
uv sync --extra cuda
```

## Quickstart

The pipeline is supervised fine-tuning, optional GRPO or OPD, then generation or
evaluation:

```bash
# 1. SFT: save PEFT adapter checkpoints in a timestamped SFT directory
uv run --extra cuda -m cuda_backend.sft_train

# 2. GRPO: continue from an SFT adapter checkpoint
uv run --extra cuda -m cuda_backend.grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_000500

# Alternatively, continue from an SFT adapter checkpoint with OPD
uv run --extra cuda -m cuda_backend.opd_train \
  --teacher-model <teacher-model> \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_000500

# 3. Generate text from a GRPO adapter checkpoint
uv run --extra cuda -m cuda_backend.generate_text \
  --model_path ./checkpoints/cuda/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter \
  --prompt "What is 2 + 2?" \
  --max-new-tokens 256

# 4. Evaluate a GRPO adapter checkpoint on the GSM8K test split
uv run --extra cuda -m cuda_backend.gsm8k_eval \
  --model_path ./checkpoints/cuda/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter
```

Generation also accepts `--tools-json` with a JSON array of available function
schemas.

## Checkpoints and monitoring

Runs are saved under `checkpoints/cuda/<algorithm>/<algorithm>_<timestamp>/`.
Each run's `run_logs/` directory contains `args.json`, `out.log`, and
TensorBoard events. Step checkpoints are saved alongside `run_logs/`.

Monitor the metrics with:

```bash
uv run --extra cuda tensorboard --logdir=./checkpoints/cuda
```

SFT, GRPO, and OPD train LoRA adapters by default. To continue SFT from an existing
PEFT adapter, provide both its base model and checkpoint:

```bash
uv run --extra cuda -m cuda_backend.sft_train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_000500
```

Pass `--train-mode full` to train and save a dense model instead. When an
adapter is also provided, it is merged into the dense model before training.

## Generation and evaluation

Generate text from a Hugging Face model or dense checkpoint:

```bash
uv run --extra cuda -m cuda_backend.generate_text \
  --model_path Qwen/Qwen2-0.5B-Instruct
```

For an SFT, GRPO, or OPD PEFT adapter checkpoint, add `--load-adapter`:

```bash
uv run --extra cuda -m cuda_backend.generate_text \
  --model_path ./checkpoints/cuda/sft/sft_<timestamp>/step_000500 \
  --load-adapter

uv run --extra cuda -m cuda_backend.gsm8k_eval \
  --model_path ./checkpoints/cuda/sft/sft_<timestamp>/step_000500 \
  --load-adapter
```

Omit `--load-adapter` when loading a Hugging Face model or dense checkpoint.

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA or full-model training
- `grpo_train.py` — group-relative policy optimization with LoRA or full-model training
- `opd_train.py` — on-policy distillation with LoRA or full-model training
- `generate_text.py` — inference from a base model, dense checkpoint, or PEFT adapter
- `gsm8k_eval.py` — greedy GSM8K test-set evaluation
- `xlam_function_calling_eval.py` — exact tool-call evaluation on the xLAM test set

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`,
`data_preparation/xlam_function_calling.py`, and the shared SFT batching in
`data_preparation/sft.py`. CUDA device placement, models, losses, and
optimization stay in these backend entrypoints.
