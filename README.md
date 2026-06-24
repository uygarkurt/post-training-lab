# Post-Training Lab — GRPO

GRPO (Group Relative Policy Optimization) training on Apple Silicon using [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm).

The main entry point is [`grpo_train_mlx.py`](grpo_train_mlx.py). It fine-tunes a language model with LoRA adapters using a reward signal derived from sampled rollouts, using a clipped surrogate objective with a KL penalty against a frozen reference model.

## Requirements

- macOS with Apple Silicon
- Python >= 3.12

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Training modes

### Debug (`--debug`)

A fast smoke test with toy prompts and a word-repetition reward. The model is rewarded for generating a target token (default: `" the"`). `--max-new-tok` is automatically set to **20**.

At the end of training, a sample generation is printed so you can inspect the policy quickly.

### Regular (default)

Trains on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (socratic split). Each step:

1. Samples a math question as a chat-formatted prompt
2. Generates `G` rollouts from the policy
3. Scores each rollout with a binary reward (1 if the extracted final answer matches ground truth, 0 otherwise)
4. Updates the policy with GRPO

Answer extraction tries `####`, phrases like "answer is X", then the last number in the output. Matching uses numeric comparison when possible. See `extract_final_answer` and `answers_match` in [`data_preperation/gsm8k_grpo.py`](data_preperation/gsm8k_grpo.py).

## LoRA

Training uses LoRA (same setup as SFT): the base model weights are frozen, LoRA adapters are applied to the last N transformer layers, and only the adapter weights are updated. The reference model used for the KL penalty is the unmodified base model.

Checkpoints are fused into full model directories via `mlx_lm.fuse`, so they load directly with `mlx_lm.load()` — no separate adapter path needed.

## Usage

```bash
# Debug smoke test
uv run python grpo_train_mlx.py --debug

# Regular GSM8K training (defaults: 100 num-iters, max_new_tok=256)
uv run python grpo_train_mlx.py

# Custom run
uv run python grpo_train_mlx.py \
  --num-iters 200 \
  --group-size 8 \
  --lr 1e-6 \
  --max-new-tok 256
```

## CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--debug` | off | Toy prompts + word-repetition reward |
| `--model` | `Qwen/Qwen2-0.5B-Instruct-MLX` | HuggingFace model name or local path |
| `--group-size` | `8` | Rollouts per prompt (G) |
| `--max-new-tok` | `256` | Max tokens per rollout (overridden to `20` in debug) |
| `--lr` | `1e-6` | AdamW learning rate |
| `--kl-coef` | `0.02` | KL penalty coefficient |
| `--clip-eps` | `0.2` | PPO clip epsilon |
| `--ppo-epochs` | `4` | Inner PPO epochs per step |
| `--num-iters` | `100` | Total gradient steps |
| `--target-word` | `" the"` | Target token for debug reward |
| `--epsilon` | `1e-8` | Advantage normalisation epsilon |
| `--lora-rank` | `8` | LoRA rank (r) |
| `--lora-alpha` | `16.0` | LoRA alpha (scale = alpha / rank) |
| `--lora-layers` | `8` | Layers to apply LoRA to (last N) |
| `--seed` | `42` | Random seed for data shuffle |
| `--val-split` | `0.1` | Fraction of GSM8K held out for validation |
| `--max-prompt-len` | `512` | Skip GSM8K prompts longer than this |
| `--eval-every` | `100` | Evaluate on validation set every N steps (`-1` to disable) |
| `--log-every` | `10` | Log tokens/sec to TensorBoard every N steps |
| `--param-log-every` | `50` | Log LoRA parameter histograms every N steps |
| `--tensorboard-dir` | `./runs/grpo` | TensorBoard log directory |
| `--save-every` | `50` | Save adapter checkpoint every N steps (`0` to disable) |
| `--checkpoint-dir` | `./checkpoints/grpo` | Checkpoint output directory |

## TensorBoard

Metrics are logged every step to `./runs/grpo`:

| Scalar | Description |
|--------|-------------|
| `train/loss` | Mean PPO loss over inner epochs |
| `train/loss_ema` | Exponential moving average of loss |
| `train/learning_rate` | Current learning rate |
| `train/tokens_per_sec` | Generated tokens per second (every `--log-every` steps) |
| `train/reward` | Mean reward across the rollout group |
| `train/advantage_mean` | Mean normalised advantage |
| `train/advantage_std` | Std of normalised advantage |
| `val/reward` | Validation answer accuracy (regular mode only) |
| `params/*` | LoRA parameter histograms (every `--param-log-every` steps) |

```bash
tensorboard --logdir=./runs/grpo
```

## Evaluation

In regular (GSM8K) mode, the held-out validation split is evaluated every `--eval-every` steps (default 100). Each val question gets one greedy rollout (`temp=0`); the metric is mean answer accuracy (same binary reward as training), logged as `val/reward` in TensorBoard.

Evaluation is skipped in debug mode. Set `--eval-every -1` to disable during GSM8K runs.

## Checkpoints

Checkpoints are saved to `./checkpoints/grpo/step_{step:06d}/` every `--save-every` steps, plus a final save at the end of training. LoRA adapters are fused into a full model directory via `mlx_lm.fuse`, loadable by mlx-lm:

```python
from mlx_lm import load, generate

model, tokenizer = load("./checkpoints/grpo/step_000050")
print(generate(model, tokenizer, prompt="I think that"))
```

## Project layout

```
grpo_train_mlx.py              # GRPO training script
data_preperation/
  gsm8k_grpo.py                # GSM8K prompt loading + answer extraction
checkpoints/grpo/              # Saved model checkpoints (gitignored)
runs/grpo/                     # TensorBoard logs (gitignored)
```
