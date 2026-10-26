# MLX Backend

The MLX backend is the runnable Apple Silicon implementation of the
post-training pipeline. Its SFT and GRPO scripts keep the training math visible
and support LoRA and full training on GSM8K and xLAM tool-calling data.

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
# 1. SFT: save native adapters and a fused model
uv run --extra mlx -m mlx_backend.sft_train \
  --checkpoint-dir ./checkpoints/mlx/sft

# 2. GRPO: continue from an SFT adapter checkpoint
uv run --extra mlx -m mlx_backend.grpo_train \
  --adapter ./checkpoints/mlx/sft/sft_<timestamp>/step_000500 \
  --checkpoint-dir ./checkpoints/mlx/grpo

# 3. Generate from a GRPO adapter checkpoint
uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter
```

The training arguments and literal defaults match their CUDA counterparts,
including the `Qwen/Qwen2.5-0.5B-Instruct` model and output paths under
`checkpoints/cuda/`. The examples above select `checkpoints/mlx/` explicitly
to keep backend outputs separate. Checkpoint formats are native to each backend;
`--adapter` accepts native MLX adapters, not CUDA PEFT adapters.

SFT uses a cosine learning-rate schedule from `5e-5` to `5e-6`, with 200
warmup steps by default. New LoRA adapters use rank 8, alpha 16, zero dropout,
and all linear layers except the output head. Training examples are shuffled
deterministically, while validation reports response-token loss before training
and every 50 steps by default.

Both trainers accept `--dataset gsm8k` (the default) or
`--dataset xlam-tool-calling-mixed`. Use `--train-mode full` to train all dense
parameters; if `--adapter` is supplied, it is merged before full training.
GRPO validates with greedy batched generation and uses the selected dataset's
binary correctness reward.

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
  --warmup-steps 0 \
  --eval-every -1 \
  --save-every 0 \
  --checkpoint-dir ./checkpoints/mlx/sft
```

## Checkpoints and monitoring

Each run is saved under
`<checkpoint-dir>/<algorithm>_<timestamp>/`. Its `run_logs/` directory contains
`args.json`, `out.log`, and TensorBoard events. Checkpoints use completed update
counts such as `step_000100/`, with a final checkpoint when needed.

Monitor the metrics with:

```bash
uv run --extra mlx tensorboard --logdir=./checkpoints/mlx
```

Each LoRA checkpoint contains `adapters.safetensors`, `adapter_config.json`,
and a fused model. Full-training checkpoints contain a dense model. Both include
the tokenizer, model configuration, and `metadata.json` with the completed step,
elapsed hours, and learning rate. Saving a fused copy leaves the live training
model unchanged.

For subsequent training, pass the original base model with `--model` and the
native adapter directory with `--adapter`. Generation retains its existing
`--load-adapter` flag; omit it to load the fused checkpoint directly:

```bash
uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/sft/sft_<timestamp>/step_000500 \
  --load-adapter

uv run --extra mlx -m mlx_backend.generate_text \
  --model_path ./checkpoints/mlx/sft/sft_<timestamp>/step_000500
```

## GRPO smoke test

Run one outer iteration over a tiny GSM8K subset using the real answer reward:

```bash
uv run --extra mlx -m mlx_backend.grpo_train \
  --debug \
  --debug-samples 2 \
  --num-iters 1 \
  --max-new-tok 16 \
  --eval-every -1 \
  --save-every 0 \
  --checkpoint-dir ./checkpoints/mlx/grpo
```

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA or full training
- `grpo_train.py` — group-relative policy optimization with LoRA or full training
- `generate_text.py` — inference from a base model, fused checkpoint, or adapter
- `gsm8k_eval.py` — GSM8K generation-accuracy validation used by GRPO
- `xlam_function_calling_eval.py` — tool-call validation used by GRPO

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
rewards come from `data_preparation/`. MLX tensor conversion,
models, losses, optimization, and checkpointing stay in these backend
entrypoints.
