# CUDA Backend

The CUDA backend is the runnable PyTorch implementation of the post-training
pipeline for NVIDIA GPUs. SFT and GRPO keep their training math visible and
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

The pipeline is supervised fine-tuning, optional GRPO, then generation or
evaluation:

```bash
# 1. SFT: save PEFT adapter checkpoints in a timestamped SFT directory
uv run --extra cuda -m cuda_backend.sft_train

# 2. GRPO: continue from an SFT adapter checkpoint
uv run --extra cuda -m cuda_backend.grpo_train \
  --model Qwen/Qwen2-0.5B-Instruct \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_000500

# 3. Evaluate a GRPO adapter checkpoint on the GSM8K test split
uv run --extra cuda -m cuda_backend.gsm8k_eval \
  --model_path ./checkpoints/cuda/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter
```

SFT uses 200 steps of warmup to `5e-5`, followed by cosine decay to `5e-6` by
default. Its LoRA configuration uses rank 8, alpha 16, zero dropout, and all
linear layers. Training examples are shuffled deterministically, while
validation reports response-token loss before training and every 50 steps
by default.

### SFT learning-rate schedule

SFT always uses Hugging Face's
[`get_cosine_with_min_lr_schedule_with_warmup`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_with_min_lr_schedule_with_warmup).
The scheduler advances after every optimizer update, independently of
`--log-every`. No scheduler-selection argument is needed.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--lr` | `5e-5` | Peak learning rate |
| `--warmup-steps` | `200` | Linear warmup to the peak; zero disables warmup |
| `--min-lr` | `5e-6` | Minimum rate at the end of cosine decay |
| `--num-iters` | `500` | Total optimizer steps, including warmup; decay ends here |

For example, add these options to an SFT command:

```bash
--num-iters 50000
```

With warmup enabled, Hugging Face initializes the learning rate at zero.
After 200 optimizer updates, the scheduler sets `5e-5` for the next update;
after 50,000 updates it reaches `5e-6`. Without warmup, the first update uses
`--lr`. These are illustrative settings, not measured optimal values.
Stopping early leaves the decay unfinished. For short runs, set
`--warmup-steps` below `--num-iters` (use zero for a one-step smoke test).

TensorBoard's `train/learning_rate` and each SFT checkpoint's `metadata.json`
record the rate used for that step's optimizer update, before the scheduler
advances. The final recorded rate can therefore be slightly above `--min-lr`.
Checkpoint metadata contains only `step`, `elapsed_hours`, and `learning_rate`;
the run's `args.json` records the training arguments. Loading an adapter starts
a new optimizer and schedule; metadata does not provide full training-state
resumption.

## Fast SFT smoke test

Run one optimizer step over two examples without validation or checkpointing:

```bash
uv run --extra cuda -m cuda_backend.sft_train \
  --debug \
  --debug-samples 2 \
  --num-iters 1 \
  --warmup-steps 0 \
  --eval-every -1 \
  --save-every 0
```

## Checkpoints and monitoring

Runs are saved under `runs/cuda/<algorithm>/<algorithm>_<timestamp>/`, with
matching checkpoints under
`checkpoints/cuda/<algorithm>/<algorithm>_<timestamp>/`. Each run directory
contains the parsed arguments, terminal output, and TensorBoard events.

Monitor the metrics with:

```bash
uv run --extra cuda tensorboard --logdir=./runs/cuda
```

SFT and GRPO train LoRA adapters by default. To continue SFT from an existing
PEFT adapter, provide both its base model and checkpoint:

```bash
uv run --extra cuda -m cuda_backend.sft_train \
  --model Qwen/Qwen2-0.5B-Instruct \
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

For an SFT or GRPO PEFT adapter checkpoint, add `--load-adapter`:

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
- `generate_text.py` — inference from a base model, dense checkpoint, or PEFT adapter
- `gsm8k_eval.py` — greedy GSM8K test-set evaluation

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`. CUDA device placement,
models, losses, and optimization stay in these backend entrypoints.
