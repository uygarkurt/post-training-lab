# Post-Training Lab

A minimal, hackable implementation of LLM post-training. Each stage, SFT and
GRPO, is a single self-contained script you can read top to bottom, understand
completely, and bend to your own dataset or reward. The priority is clarity over
abstraction: no framework to learn, no layers of indirection, just the algorithm in
front of you. Because each script stands alone, it doubles as an experimentation
surface: drop in your own algorithm, or change the existing loss, advantage, or
reward, and immediately see how behavior shifts. No need to trace how a dozen files
wire together. LoRA keeps it runnable on modest hardware. The current backend is
[MLX](https://github.com/ml-explore/mlx) (Apple Silicon); a CUDA backend is planned.

## Supported Algorithms


| Algorithm | MLX (Apple Silicon) | CUDA |
| --------- | ------------------- | ---- |
| SFT       | ✅                   | ❌    |
| GRPO      | ✅                   | ❌    |


## Requirements

- macOS with Apple Silicon
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

## Quickstart

The pipeline is three scripts — supervised fine-tuning, then GRPO, then generation:

```bash
# 1. SFT — LoRA fine-tune on GSM8K (saves fused model + adapters to checkpoints/sft/)
uv run python sft_train_mlx.py

# 2. GRPO — continue LoRA training from the SFT adapters with a reward signal
uv run python grpo_train_mlx.py --model ./checkpoints/sft/step_000500 --load-adapter

# 3. Generate from a checkpoint
uv run python generate_text_mlx.py --model_path ./checkpoints/grpo/step_000050 --load-adapter
```

Fast smoke test (no SFT needed) — overfit a tiny GSM8K subset with the real
answer-matching reward; watch `train/reward` and `val/reward` climb to ~1.0:

```bash
uv run python grpo_train_mlx.py --debug --lr 1e-5 --eval-every 10 --num-iters 200
```

## Monitoring

Metrics are logged to TensorBoard (`./runs/grpo`, `./runs/sft`):

```bash
tensorboard --logdir=./runs
```

## Project layout

```
post-training-lab/
│
├── sft_train_mlx.py         # SFT training (LoRA)
├── grpo_train_mlx.py        # GRPO training (LoRA)
├── generate_text_mlx.py     # Inference from a base model or checkpoint
│
├── data_preparation/
│   ├── gsm8k.py             # GSM8K dataloaders for SFT
│   └── gsm8k_grpo.py        # GSM8K prompts + answer extraction/matching for GRPO
│
├── checkpoints/             # Saved checkpoints        (gitignored)
└── runs/                    # TensorBoard logs         (gitignored)
```

## License

[MIT](LICENSE)