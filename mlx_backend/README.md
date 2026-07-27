# MLX Backend

The MLX backend is the runnable Apple Silicon implementation of the
post-training pipeline. Its training scripts keep the mathematical steps
visible and use LoRA so experiments can run on modest hardware.

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

The pipeline is supervised fine-tuning, GRPO, then generation:

```bash
# 1. SFT: saves fused model + adapters under checkpoints/mlx/sft/
uv run --extra mlx -m mlx_backend.sft_train

# 2. GRPO: continue from an SFT adapter checkpoint
uv run --extra mlx -m mlx_backend.grpo_train \
  --model ./checkpoints/mlx/sft/step_000500 \
  --load-adapter

# 3. Generate from a GRPO adapter checkpoint
uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/grpo/step_000050 \
  --load-adapter
```

## Fast smoke test

Overfit a tiny GSM8K subset using the real answer-matching reward:

```bash
uv run --extra mlx -m mlx_backend.grpo_train \
  --debug \
  --lr 1e-5 \
  --eval-every 10 \
  --num-iters 200
```

## Monitoring

Metrics are logged under `./runs/mlx/`:

```bash
uv run --extra mlx tensorboard --logdir=./runs/mlx
```

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA
- `grpo_train.py` — group-relative policy optimization with LoRA
- `generate_text.py` — inference from a base model or adapter checkpoint

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`. MLX tensor conversion,
training, and checkpointing stay in these backend entrypoints.
