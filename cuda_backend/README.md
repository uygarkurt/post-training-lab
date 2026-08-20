# CUDA Backend

CUDA support for NVIDIA GPUs is in progress. Text generation and GRPO are
currently implemented.

The available entrypoints are:

- `sft_train.py` (CUDA SFT scaffold with an exposed training loop)
- `grpo_train.py`
- `generate_text.py`
- `gsm8k_eval.py`

Install the CUDA dependencies from the repository root:

```bash
uv sync --extra cuda
```

Build on the CUDA SFT scaffold, then run it with:

```bash
uv run --extra cuda -m cuda_backend.sft_train
```

Train the default model with GRPO and LoRA:

```bash
uv run --extra cuda -m cuda_backend.grpo_train
```

To continue training from an existing PEFT adapter checkpoint:

```bash
uv run --extra cuda -m cuda_backend.grpo_train \
  --model Qwen/Qwen2-0.5B-Instruct \
  --adapter ./checkpoints/cuda/grpo_<timestamp>/step_000100
```

Generate from a Hugging Face model:

```bash
uv run --extra cuda -m cuda_backend.generate_text
```

To generate from a PEFT adapter checkpoint, pass its directory:

```bash
uv run --extra cuda -m cuda_backend.generate_text \
  --model_path ./checkpoints/cuda/grpo_<timestamp>/step_000100 \
  --load-adapter
```

Evaluate a PEFT adapter checkpoint on the GSM8K test split:

```bash
uv run --extra cuda -m cuda_backend.gsm8k_eval \
  --model_path ./checkpoints/cuda/grpo_<timestamp>/step_000100 \
  --load-adapter
```

Omit `--load-adapter` when evaluating a full-model checkpoint.

CUDA entrypoints consume the same backend-neutral samples and answer
matching functions from `data_preparation/gsm8k.py`, including its PyTorch
DataLoaders. CUDA device placement, models, losses, and optimization steps will
remain in the CUDA scripts.
