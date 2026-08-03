# CUDA Backend

CUDA support for NVIDIA GPUs is in progress. Text generation and GRPO are
currently implemented.

The available entrypoints are:

- `grpo_train.py`
- `generate_text.py`

Install the CUDA dependencies from the repository root:

```bash
uv sync --extra cuda
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

CUDA entrypoints consume the same backend-neutral samples and answer
matching functions from `data_preparation/gsm8k.py`, including its PyTorch
DataLoaders. CUDA device placement, models, losses, and optimization steps will
remain in the CUDA scripts.
