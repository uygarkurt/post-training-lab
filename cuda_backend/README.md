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
  --dataset xlam-function-calling-irrelevance \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_000500

# 3. Evaluate a GRPO adapter checkpoint on the xLAM test split
uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path ./checkpoints/cuda/grpo/grpo_<timestamp>/step_000500 \
  --load-adapter
```

SFT, GRPO, and evaluation load
[`uygarkurt/xlam-tool-calling-mixed`](https://huggingface.co/datasets/uygarkurt/xlam-tool-calling-mixed)
directly from Hugging Face; no local preparation is needed. SFT and GRPO use
the `train` split, and evaluation uses `test`. The binary
reward checks exact tool-call sets for positive examples and correct abstention
for irrelevance examples. Validation reports overall exact and function-name
accuracy plus separate call and no-call accuracy when applicable. Use
`--dataset gsm8k` to select the original numeric-answer dataset and reward.

SFT uses 200 steps of warmup to `5e-5`, followed by cosine decay to `5e-6` by
default. Its LoRA configuration uses rank 8, alpha 16, zero dropout, and all
linear layers. Training examples are shuffled deterministically, while
validation reports response-token loss before training and every 50 steps
by default. The default dataset is the hosted xLAM and irrelevance split,
with a 512-token sequence limit.

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
Each SFT and GRPO checkpoint stores only `step`, `elapsed_hours`, and
`learning_rate` in `metadata.json`; the run's `run_logs/args.json` records the training
arguments. In both trainers, the elapsed timer starts immediately before the
optimization loop, after initial validation. Loading an adapter starts a new
optimizer; metadata does not provide full training-state resumption.

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

Runs are saved under `checkpoints/cuda/<algorithm>/<algorithm>_<timestamp>/`.
Each run's `run_logs/` directory contains `args.json`, `out.log`, and
TensorBoard events. Step checkpoints are saved alongside `run_logs/`.

Monitor the metrics with:

```bash
uv run --extra cuda tensorboard --logdir=./checkpoints/cuda
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

Evaluate the xLAM/irrelevance mixture:

```bash
uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 100 \
  --output-jsonl xlam_base.jsonl
```

The hosted dataset converts the source's JSON-encoded tool definitions to standard
Transformers schemas and creates query-group-disjoint 62,485/3,289 train/test
splits containing calls and irrelevant-tool no-call examples. Evaluation
defaults to this mixed test split, supplies tools through the tokenizer's native
chat template, and reports strict call-set accuracy, function-name-set accuracy, parse counts,
and truncations. Parallel-call order and JSON object key order are ignored;
names, argument values and types, array order, missing calls, and extra calls
remain significant. See the
[dataset card](https://huggingface.co/datasets/uygarkurt/xlam-tool-calling-mixed) for the complete
recipe and limitations.

To evaluate a local prepared test JSONL file instead, pass `--dataset-path <file>`.

Train GRPO on the hosted xLAM and irrelevance training pool, optionally
initialized from an SFT adapter:

```bash
uv run --extra cuda -m cuda_backend.grpo_train \
  --dataset xlam-function-calling-irrelevance \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter ./checkpoints/cuda/sft/sft_<timestamp>/step_<number>
```

The mixed training split contains 56,753 call examples and 5,732 irrelevant-tool
no-call examples. With the pinned Qwen tokenizer, the default 640-token prompt
limit retains 43,928 calls and 4,863 no-calls before the seeded 5% runtime
validation split. Overlong prompts are dropped whole. GRPO uses a binary
reward with the following contract:

- Predictions may contain Qwen-style `<tool_call>` blocks or a bare JSON
  object or list, optionally wrapped in a complete JSON code fence.
- Calls are normalized to a function name and argument object. JSON object key
  order and parallel-call order do not matter; JSON types, values, array order,
  duplicate calls, missing calls, and extra calls do matter.
- A call-required rollout receives 1 only for an exact normalized call-multiset
  match. Malformed, partial, extra, or missing calls receive 0.
- A no-call rollout receives 1 for literal `[]` or ordinary non-tool-call text.
  A parsed call, attempted `<tool_call` tag, or other JSON-like attempted call
  receives 0.
- There is no format, function-name, or partial-argument reward shaping.

Training and evaluation use the same binary exact correctness reward.

TensorBoard logs `train/all_zero_group_fraction`,
`train/all_one_group_fraction`, and `train/mixed_group_fraction` over each
`--log-every` interval (10 prompt groups by default). These are fractions of
groups, not individual completions or inner optimizer updates. They sum to 1.
Counters reset after each log point; the final point includes any remaining
steps using their actual group count.

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA or full-model training
- `grpo_train.py` — group-relative policy optimization with LoRA or full-model training
- `generate_text.py` — inference from a base model, dense checkpoint, or PEFT adapter
- `gsm8k_eval.py` — greedy GSM8K test-set evaluation
- `xlam_function_calling_eval.py` — exact tool-call evaluation on the xLAM test set

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`,
`data_preparation/xlam_function_calling.py`, and the shared SFT batching in
`data_preparation/sft.py`. CUDA device placement, models, losses, and
optimization stay in these backend entrypoints.
