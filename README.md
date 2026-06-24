# Post-Training Lab — GRPO

GRPO (Group Relative Policy Optimization) training on Apple Silicon using [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm).

The main entry point is [`grpo_train_mlx.py`](grpo_train_mlx.py). It fine-tunes a language model with a reward signal derived from sampled rollouts, using a clipped surrogate objective with a KL penalty against a frozen reference model.

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
3. Scores each rollout with a binary reward (1 if the final numeric answer matches ground truth, 0 otherwise)
4. Updates the policy with GRPO

Ground-truth answers are parsed from the `####` format. Data loading lives in [`data_preperation/gsm8k_grpo.py`](data_preperation/gsm8k_grpo.py).

## Usage

```bash
# Debug smoke test
uv run python grpo_train_mlx.py --debug

# Regular GSM8K training (defaults: 100 steps, max_new_tok=256)
uv run python grpo_train_mlx.py

# Custom run
uv run python grpo_train_mlx.py \
  --steps 200 \
  --group-size 8 \
  --lr 1e-6 \
  --max-new-tok 256
```

## CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--debug` | off | Toy prompts + word-repetition reward |
| `--model` | `Qwen/Qwen2-0.5B-Instruct` | HuggingFace model name or local path |
| `--group-size` | `8` | Rollouts per prompt (G) |
| `--max-new-tok` | `256` | Max tokens per rollout (overridden to `20` in debug) |
| `--lr` | `1e-6` | AdamW learning rate |
| `--kl-coef` | `0.02` | KL penalty coefficient |
| `--clip-eps` | `0.2` | PPO clip epsilon |
| `--ppo-epochs` | `4` | Inner PPO epochs per step |
| `--steps` | `100` | Total training steps |
| `--target-word` | `" the"` | Target token for debug reward |
| `--epsilon` | `1e-8` | Advantage normalisation epsilon |
| `--seed` | `42` | Random seed for data shuffle |
| `--val-split` | `0.1` | Fraction of GSM8K held out (not used during training) |
| `--max-prompt-len` | `512` | Skip GSM8K prompts longer than this |
| `--tensorboard-dir` | `./runs/grpo` | TensorBoard log directory |
| `--save-steps` | `50` | Checkpoint every N steps (`0` to disable) |
| `--checkpoint-dir` | `./checkpoints/grpo` | Checkpoint output directory |

## TensorBoard

Metrics are logged every step to `./runs/grpo`:

| Scalar | Description |
|--------|-------------|
| `train/loss` | Mean PPO loss over inner epochs |
| `train/reward` | Mean reward across the rollout group |
| `train/reward_avg10` | 10-step moving average of reward |
| `train/advantage_mean` | Mean normalised advantage |
| `train/advantage_std` | Std of normalised advantage |

```bash
tensorboard --logdir=./runs/grpo
```

## Checkpoints

Checkpoints are saved to `./checkpoints/grpo/step_{step:06d}/` every `--save-steps` steps, plus a final save at the end of training. Each checkpoint is a full model directory loadable by mlx-lm:

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
