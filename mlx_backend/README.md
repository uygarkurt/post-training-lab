# MLX Backend

The MLX backend is the runnable Apple Silicon implementation of the
post-training pipeline. Its SFT and GRPO scripts keep the training math visible
and use LoRA so experiments can run on modest hardware.

Run all commands below from the repository root.

## Requirements

- macOS with Apple Silicon
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

Install the shared project dependencies and the MLX extra:

```bash
uv sync --extra mlx
```

## Quickstart

The pipeline is supervised fine-tuning, optional GRPO, then generation:

```bash
# 1. SFT: save fused model and adapter files in a timestamped SFT directory
uv run --extra mlx -m mlx_backend.sft_train

# 2. GRPO: continue from an SFT adapter checkpoint
uv run --extra mlx -m mlx_backend.grpo_train \
  --model ./checkpoints/mlx/sft/sft_<timestamp>/step_000500 \
  --load-adapter

# 3. Generate from a GRPO adapter checkpoint
uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter
```

SFT uses a constant `5e-5` learning rate by default. Its LoRA configuration
uses rank 8, alpha 16, zero dropout, and all transformer layers. Training
examples are shuffled deterministically, while validation reports
response-token loss before training and every 50 steps by default.

The MLX training behavior matches CUDA where the backends share semantics:
AdamW uses bias correction and weight decay, gradients are clipped to global
norm 1.0, and metrics and checkpoints use the same step numbering and cadence.

## Fast SFT smoke test

Run one optimizer step over two examples without validation or checkpointing:

```bash
uv run --extra mlx -m mlx_backend.sft_train \
  --debug \
  --debug-samples 2 \
  --num-iters 1 \
  --eval-every -1 \
  --save-every 0
```

## Checkpoints and monitoring

Runs are saved under `runs/mlx/<algorithm>/<algorithm>_<timestamp>/`, with
matching checkpoints under
`checkpoints/mlx/<algorithm>/<algorithm>_<timestamp>/`. Each run directory
contains the parsed arguments, terminal output, and TensorBoard events.

Monitor the metrics with:

```bash
uv run --extra mlx tensorboard --logdir=./runs/mlx
```

Each SFT checkpoint contains both `adapters.safetensors` with its
`adapter_config.json` and a fused model produced by `mlx_lm.fuse`. Load the
adapter representation with `--load-adapter`, or omit that flag to load the
fused checkpoint directly:

```bash
uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/sft/sft_<timestamp>/step_000500 \
  --load-adapter

uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/sft/sft_<timestamp>/step_000500
```

Set `--lora-layers N` to restrict LoRA to the final `N` transformer layers;
the default `-1` applies it to every layer.

## GRPO smoke test

Overfit a tiny GSM8K subset using the real answer-matching reward:

```bash
uv run --extra mlx -m mlx_backend.grpo_train \
  --debug \
  --lr 1e-5 \
  --eval-every 10 \
  --num-iters 200
```

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA
- `grpo_train.py` — group-relative policy optimization with LoRA
- `generate_text.py` — inference from a base model, fused checkpoint, or adapter
- `gsm8k_eval.py` — GSM8K generation-accuracy validation used by GRPO

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`. MLX tensor conversion,
models, losses, optimization, and checkpointing stay in these backend
entrypoints.
