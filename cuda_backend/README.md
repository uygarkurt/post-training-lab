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
by default. The default dataset is the prepared xLAM function-calling split,
with a 2,048-token sequence limit.

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

Evaluate a NuminaMath Algebra checkpoint on the local
`data/numinamath-1.5-rl-verifiable/test.jsonl` file:

```bash
uv run --extra cuda --extra eval -m cuda_backend.numinamath_eval \
  --model_path ./checkpoints/cuda/sft/sft_<timestamp>/step_000500 \
  --load-adapter \
  --batch-size 8 \
  --max-prompt-len 512 \
  --max-new-tokens 512 \
  --output-jsonl numinamath_sft.jsonl
```

This follows GSM8K evaluation's model loading, greedy batched generation,
progress bar, and final accuracy output. The prompt is the same problem-only
user message used for NuminaMath SFT. Prompts are left-padded; prompts longer
than `--max-prompt-len` are skipped and reported, never truncated.

The default baseline is `Qwen/Qwen2.5-0.5B-Instruct`, matching CUDA SFT.
Compare an adapter against the exact base model named in its
`adapter_config.json`; Qwen2 and Qwen2.5 are different baselines. For example:

```bash
uv run --extra cuda --extra eval -m cuda_backend.numinamath_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 1000 \
  --output-jsonl numinamath_base.jsonl
```

Evaluation reports how many completions reach `--max-new-tokens` without an
end-of-sequence token. `--output-jsonl` optionally saves questions, original
and parsed references, generated completions, correctness, and truncation
flags for inspection. The output file is overwritten on each invocation.

Add `--num-samples 1000` to evaluate the first 1,000 usable examples in file
order, after filtering overlong prompts and unparseable references. The local
test file is already shuffled, so this selects a repeatable subset. Omitting
the option evaluates all usable examples; requesting more than are available
uses all of them. Dataset loading and reference parsing still cover the full
file; the limit reduces model generation. Keep the same sample limit when
comparing checkpoints.

Answer matching uses [Math-Verify](https://github.com/huggingface/Math-Verify)
from the existing `eval` extra to compare mathematical expressions against
the source `answer` field. Boxed answers take priority, with ordinary LaTeX
and numeric answers also supported. Rows with unparseable reference answers
are discarded from evaluation and reported in the skipped count. Unparseable
model outputs count as incorrect. String fallback is disabled for both
references and model outputs: an extracted string alone is not sufficient.
Accuracy measures automatic answer agreement over the retained subset;
parseable but incomplete source answers can still affect the score. Keep generation
limits fixed when comparing checkpoints. Only the local test file is loaded;
training and runtime validation continue to use `train.jsonl`.

The source solutions are free-form and do not consistently mark a final answer.
Grading the first 1,000 usable source solutions with this evaluator and
Math-Verify 0.9.0 accepts only 485: extraction failures and incomplete or noisy
reference answers affect even the supplied solutions. Treat this score as
automatic answer agreement, and inspect saved completions before interpreting
a change as improved or degraded reasoning. Comparison timeouts count as
incorrect; they do not stop evaluation.

Evaluate MetaMathQA using the prepared local test file:

```bash
uv run --extra eval python data/metamathqa/prepare.py
uv run --extra cuda --extra eval -m cuda_backend.metamathqa_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-path data/metamathqa/test.jsonl \
  --num-samples 100 \
  --output-jsonl metamathqa_base.jsonl
```

Preparation deduplicates exact queries by default; add
`--keep-duplicate-queries` to retain them. Unparseable references are removed
before splitting, using Math-Verify from the `eval` extra. MetaMathQA evaluation
follows the NuminaMath CLI and generation settings, using only `query` as the
prompt and
the extracted `ground_truth` as the reference. If a completion contains
`The answer is:`, its final suffix is graded; otherwise boxed/LaTeX/numeric
extraction is used. See the [data README](../data/metamathqa/README.md) for
split counts, filtering, and answer-matching behavior.

Prepare and evaluate the xLAM function-calling dataset:

```bash
uv run python data/xlam-function-calling-60k/prepare.py
uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 100 \
  --output-jsonl xlam_base.jsonl
```

Preparation converts the source's JSON-encoded tool definitions to standard
Transformers schemas and creates query-group-disjoint 56,753/2,987 train/test
splits. Evaluation supplies tools through the tokenizer's native chat template
and reports strict call-set accuracy, function-name-set accuracy, parse counts,
and truncations. Parallel-call order and JSON object key order are ignored;
names, argument values and types, array order, missing calls, and extra calls
remain significant. See the
[xLAM data README](../data/xlam-function-calling-60k/README.md) for the complete
recipe and limitations.

## Entrypoints

- `sft_train.py` — supervised fine-tuning with LoRA or full-model training
- `grpo_train.py` — group-relative policy optimization with LoRA or full-model training
- `generate_text.py` — inference from a base model, dense checkpoint, or PEFT adapter
- `gsm8k_eval.py` — greedy GSM8K test-set evaluation
- `numinamath_eval.py` — greedy batched evaluation on the local NuminaMath Algebra test set
- `metamathqa_eval.py` — greedy batched evaluation on the local MetaMathQA test set
- `xlam_function_calling_eval.py` — exact tool-call evaluation on the local xLAM test set

Dataset loading, tokenization, splitting, PyTorch DataLoaders, padding, and
answer matching come from `data_preparation/gsm8k.py`,
`data_preparation/numinamath.py`, `data_preparation/metamathqa.py`,
`data_preparation/xlam_function_calling.py`, and the shared SFT batching in
`data_preparation/sft.py`. CUDA device placement, models, losses, and
optimization stay in these backend entrypoints.
