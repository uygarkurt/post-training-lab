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

A fast algorithm smoke test that **overfits a tiny GSM8K subset** using the real answer-matching reward. By default **8 samples** (`--debug-samples`) are drawn from GSM8K; the **same set is used for training and validation**, so you should see `train/reward` and `val/reward` climb as the policy memorises those questions.

Recommended starting command:

```bash
uv run python grpo_train_mlx.py --debug --lr 1e-5 --eval-every 10 --num-iters 200
```

At the end of training, a sample generation is printed for the first debug question.

For a minimal word-repetition toy (no GSM8K), use [`scripts/grpo_train_mlx_toy.py`](scripts/grpo_train_mlx_toy.py).

### Regular (default)

Trains on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (socratic split). Each step:

1. Samples a math question as a chat-formatted prompt
2. Generates `G` rollouts from the policy
3. Scores each rollout with a binary reward (1 if the extracted final answer matches ground truth, 0 otherwise)
4. Updates the policy with GRPO

Answer extraction tries `####`, phrases like "answer is X", then the last number in the output. Matching uses numeric comparison when possible. See `extract_final_answer` and `answers_match` in [`data_preperation/gsm8k_grpo.py`](data_preperation/gsm8k_grpo.py).

## LoRA

Training uses LoRA (same setup as SFT): the base model weights are frozen, LoRA adapters are applied to the last N transformer layers, and only the adapter weights are updated.

- **Without `--load-adapter`:** the reference model for the KL penalty is the unmodified base model.
- **With `--load-adapter`:** the reference model is the SFT policy (base + loaded adapters, frozen).

Checkpoints are fused into full model directories via `mlx_lm.fuse`, so they load directly with `mlx_lm.load()` for inference. Each checkpoint also keeps `adapters.safetensors` and `adapter_config.json` on disk for continued LoRA training.

## SFT → GRPO workflow

Run SFT first ([`sft_train_mlx.py`](sft_train_mlx.py)); checkpoints are saved under `./checkpoints/sft/step_{step}/` with both a fused model and adapter files.

Then resume those adapters in GRPO by pointing `--model` at the SFT step directory and enabling `--load-adapter`:

```bash
# 1. SFT (saves fused model + adapters to checkpoints/sft/)
uv run python sft_train_mlx.py

# 2. GRPO — continue LoRA training from SFT adapters
uv run python grpo_train_mlx.py \
  --model ./checkpoints/sft/step_000500 \
  --load-adapter
```

When `--load-adapter` is set, `--model` must be an SFT checkpoint directory containing `adapters.safetensors` and `adapter_config.json`. The unfused base model is read from `base_model` in that config; LoRA hyperparameters are taken from the adapter config, not CLI `--lora-*` flags.

## Usage

```bash
# Debug overfit smoke test (tiny GSM8K subset, real reward)
uv run python grpo_train_mlx.py --debug --lr 1e-5 --eval-every 10 --num-iters 200

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
| `--debug` | off | Overfit tiny GSM8K subset (same samples for train and val) |
| `--debug-samples` | `8` | GSM8K samples used in `--debug` mode |
| `--model` | `Qwen/Qwen2-0.5B-Instruct-MLX` | HuggingFace model name or path; with `--load-adapter`, path to an SFT step directory |
| `--load-adapter` | off | Resume LoRA from SFT adapters in `--model` directory |
| `--group-size` | `8` | Rollouts per prompt (G) |
| `--max-new-tok` | `256` | Max tokens per rollout |
| `--lr` | `1e-6` | AdamW learning rate |
| `--kl-coef` | `0.02` | KL penalty coefficient |
| `--clip-eps` | `0.2` | PPO clip epsilon |
| `--ppo-epochs` | `4` | Inner PPO epochs per step |
| `--num-iters` | `100` | Total gradient steps |
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

In regular (GSM8K) mode, the held-out validation split is evaluated every `--eval-every` steps (default 100). In `--debug` mode, validation runs on the same tiny overfit set. Each val question gets one greedy rollout (`temp=0`); the metric is mean answer accuracy (same binary reward as training), logged as `val/reward` in TensorBoard.

Set `--eval-every -1` to disable validation.

## Checkpoints

Checkpoints are saved to `./checkpoints/grpo/step_{step:06d}/` every `--save-every` steps, plus a final save at the end of training. Each directory contains a fused model (loadable by mlx-lm) plus `adapters.safetensors` and `adapter_config.json` for continued LoRA training:

```python
from mlx_lm import load, generate

model, tokenizer = load("./checkpoints/grpo/step_000050")
print(generate(model, tokenizer, prompt="I think that"))
```

## Project layout

```
sft_train_mlx.py               # SFT training script (checkpoints/sft/)
grpo_train_mlx.py              # GRPO training script
data_preperation/
  gsm8k_grpo.py                # GSM8K prompt loading + answer extraction
checkpoints/sft/               # SFT checkpoints (gitignored)
checkpoints/grpo/              # GRPO checkpoints (gitignored)
runs/                          # TensorBoard logs (gitignored)
```
