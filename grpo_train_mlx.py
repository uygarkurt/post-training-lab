import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tensorboardX import SummaryWriter
from tqdm import tqdm

import data_preperation.gsm8k_grpo as gsm8k_grpo


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training on MLX (Apple Silicon).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Overfit a tiny GSM8K subset (same samples for train and val; real answer reward)",
    )
    parser.add_argument(
        "--debug-samples", type=int, default=8,
        help="Number of GSM8K samples in --debug mode (train and val use the same set)",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct-MLX", help="HuggingFace model name or path")
    parser.add_argument(
        "--load-adapter", action="store_true",
        help="Resume LoRA from SFT adapters in --model directory (requires adapters.safetensors + adapter_config.json)",
    )

    parser.add_argument("--group-size", type=int, default=8, help="Number of rollouts per prompt (G)")
    parser.add_argument("--max-new-tok", type=int, default=512, help="Max tokens to generate per rollout")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO inner epochs per step")
    parser.add_argument("--num-iters", type=int, default=500, help="Total gradient steps")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Advantage normalisation epsilon")

    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (scale = alpha / rank)")
    parser.add_argument("--lora-layers", type=int, default=8, help="Number of transformer layers to apply LoRA to (last N)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffle")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction held out from GSM8K train set")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Skip GSM8K prompts longer than this")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate on validation set every N steps (-1 to disable)")

    parser.add_argument("--log-every", type=int, default=10, help="Log tokens/sec to TensorBoard every N steps")
    parser.add_argument("--param-log-every", type=int, default=50, help="Log LoRA parameter histograms every N steps")

    parser.add_argument("--tensorboard-dir", type=str, default="./runs/grpo", help="Directory for TensorBoard logs")

    parser.add_argument("--save-every", type=int, default=100, help="Save adapter checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/grpo", help="Directory for checkpoints")

    args = parser.parse_args()
    return args


def gsm8k_answer_reward(trajectory_tokens, masks, tokenizer, ground_truth):
    rewards = []
    for g in range(trajectory_tokens.shape[0]):
        valid_ids = [
            int(t) for t, m in zip(trajectory_tokens[g].tolist(), masks[g].tolist())
            if m == 1
        ]
        text = tokenizer.decode(valid_ids)
        pred = gsm8k_grpo.extract_final_answer(text)
        rewards.append(1.0 if gsm8k_grpo.answers_match(pred, ground_truth) else 0.0)
    return mx.array(rewards, dtype=mx.float32)


def batched_rollout(model, prompt_tokens, group_size, max_new_tok, sampler, tokenizer):
    prompt_batch = mx.repeat(prompt_tokens[None, :], group_size, axis=0) # [G, P]

    cache = make_prompt_cache(model)
    logits = model(prompt_batch, cache=cache)[:, -1, :] # [G, V]

    eos_id = tokenizer.eos_token_id
    active = mx.ones((group_size,), dtype=mx.bool_)

    tokens_steps = []
    logprobs_steps = []
    masks_steps = []
    for _ in range(max_new_tok):
        lp = nn.log_softmax(logits, axis=-1)
        next_tok = sampler(lp)  # [G]
        tok_logprob = lp[mx.arange(group_size), next_tok]   # [G]

        masks_steps.append(active.astype(mx.int32))
        tokens_steps.append(next_tok)
        logprobs_steps.append(tok_logprob)

        active = active & (next_tok != eos_id)
        logits = model(next_tok[:, None], cache=cache)[:, -1, :]  # [G, V]

    tokens = mx.stack(tokens_steps, axis=-1)    # [G, L]
    logprobs = mx.stack(logprobs_steps, axis=-1)  # [G, L]
    masks = mx.stack(masks_steps, axis=-1)     # [G, L]

    tokens = mx.where(masks == 1, tokens, tokenizer.pad_token_id)
    logprobs = logprobs * masks
    return tokens, logprobs, masks


def token_logprobs(model, prompt_tokens, input_ids, loss_mask):
    G = input_ids.shape[0]
    P = prompt_tokens.shape[0]

    prompt_batch = mx.repeat(prompt_tokens[None, :], G, axis=0) # [G, P]
    full_ids = mx.concatenate([prompt_batch, input_ids], axis=-1) # [G, L+P]

    logits = model(full_ids) # [G, L+P, V]
    logprobs = nn.log_softmax(logits, axis=-1) # [G, L+P, V]

    gen_logprobs = logprobs[:, P-1:-1, :] # [G, L, V]

    g_ids = mx.expand_dims(mx.arange(G), axis=-1) # [G, 1]
    l_idx = mx.expand_dims(mx.arange(input_ids.shape[1]), 0) # [1, L]
    final_logprobs = gen_logprobs[g_ids, l_idx, input_ids] # [G, L]

    return final_logprobs * loss_mask


def evaluate(model, val_samples, tokenizer, args):
    """
    Compute mean reward (answer accuracy) over the validation set (no gradients).

    Uses greedy decoding (temp=0) with a single rollout per question.
    """
    if not val_samples:
        return float("nan")

    sampler = make_sampler(temp=0.0)
    total_reward = 0.0

    for sample in tqdm(val_samples, desc="  eval", leave=False, unit="sample"):
        prompt_tokens = mx.array(sample["prompt_ids"])
        tokens, _, masks = batched_rollout(
            model,
            prompt_tokens,
            group_size=1,
            max_new_tok=args.max_new_tok,
            sampler=sampler,
            tokenizer=tokenizer,
        )
        reward = gsm8k_answer_reward(tokens, masks, tokenizer, sample["ground_truth"])
        total_reward += reward.item()

    return total_reward / len(val_samples)


def read_adapter_config(adapter_dir):
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"--load-adapter requires adapter_config.json in {adapter_dir}"
        )
    with open(config_path) as f:
        return json.load(f)


def apply_adapter_config_to_args(args, adapter_cfg):
    """Sync LoRA hyperparameters from adapter_config.json onto args."""
    lora_params = adapter_cfg["lora_parameters"]
    saved_layers = adapter_cfg["num_layers"]
    saved_rank = lora_params["rank"]
    saved_alpha = lora_params["scale"] * saved_rank

    if (args.lora_layers, args.lora_rank, args.lora_alpha) != (saved_layers, saved_rank, saved_alpha):
        print(
            "  Note: using LoRA config from adapter_config.json "
            f"(layers={saved_layers}, rank={saved_rank}, alpha={saved_alpha}); "
            "CLI --lora-* values ignored."
        )

    args.lora_layers = saved_layers
    args.lora_rank = saved_rank
    args.lora_alpha = saved_alpha


def save_full_checkpoint(model, step, args):
    """
    Save a fully-merged checkpoint loadable by mlx_lm.load().

    Writes adapter weights + adapter_config.json into the checkpoint directory,
    then calls mlx_lm.fuse to produce a complete model directory. Adapters are
    kept on disk for continued LoRA training.
    """
    out_dir = os.path.join(args.checkpoint_dir, f"step_{step:06d}")
    os.makedirs(out_dir, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(out_dir, "adapters.safetensors"), adapter_weights)

    adapter_cfg = {
        "fine_tune_type": "lora",
        "base_model": args.base_model,
        "num_layers": args.lora_layers,
        "lora_parameters": {
            "rank":    args.lora_rank,
            "scale":   args.lora_alpha / args.lora_rank,
            "dropout": 0.05,
        },
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_cfg, f, indent=2)

    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm.fuse",
         "--model",        args.base_model,
         "--adapter-path", out_dir,
         "--save-path",    out_dir],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"  Warning: mlx_lm.fuse failed (step {step}):")
        print(result.stderr.strip())
    else:
        print(f"  checkpoint -> {out_dir}/  "
              f"(load with: mlx_lm.load('{out_dir}'))")


def main():
    args = parse_args()

    # ---- Load model --------------------------------------------------------
    if args.load_adapter:
        adapter_dir = args.model
        adapters_file = os.path.join(adapter_dir, "adapters.safetensors")
        if not os.path.isfile(adapters_file):
            raise FileNotFoundError(
                f"--load-adapter requires adapters.safetensors in {adapter_dir}"
            )

        adapter_cfg = read_adapter_config(adapter_dir)
        base_model = adapter_cfg.get("base_model")
        if not base_model:
            raise ValueError(
                f"adapter_config.json in {adapter_dir} is missing 'base_model'"
            )

        args.base_model = base_model
        apply_adapter_config_to_args(args, adapter_cfg)

        print(f"Loading base model {base_model} ...")
        policy, tokenizer = load(base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy.freeze()
        print(f"Loading SFT adapters from {adapter_dir} ...")
        load_adapters(policy, adapter_dir)
    else:
        args.base_model = args.model
        print(f"Loading {args.model} ...")
        policy, tokenizer = load(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy.freeze()
        lora_cfg = {
            "rank":    args.lora_rank,
            "scale":   args.lora_alpha / args.lora_rank,
            "dropout": 0.05,
        }
        linear_to_lora_layers(policy, num_layers=args.lora_layers, config=lora_cfg)

    trainable = list(tree_flatten(policy.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    print(f"LoRA applied: {n_trainable:,} trainable params across {len(trainable)} tensors "
          f"({args.lora_layers} layers, r={args.lora_rank})")

    if args.load_adapter:
        ref, _ = load(args.base_model)
        ref.freeze()
        load_adapters(ref, args.model)
    else:
        ref, _ = load(args.model)

    # ---- Load data ---------------------------------------------------------
    print("Loading dataset ...")
    if args.debug:
        gsm8k_train, val_samples = gsm8k_grpo.build_debug_overfit_samples(tokenizer, args)
    else:
        gsm8k_train, val_samples = gsm8k_grpo.build_grpo_samples(tokenizer, args)
    data_iter = itertools.cycle(gsm8k_train)

    # ---- Optimizer ---------------------------------------------------------
    optimizer = optim.AdamW(learning_rate=args.lr)

    # ---- TensorBoard -------------------------------------------------------
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs -> {args.tensorboard_dir}  (run: tensorboard --logdir={args.tensorboard_dir})")

    # ---- Training loop -----------------------------------------------------
    mode = "debug" if args.debug else "gsm8k"
    print(f"\nTraining: {args.num_iters} steps | group={args.group_size} | lr={args.lr} | "
          f"lora_r={args.lora_rank} | mode={mode}\n")

    step     = 0
    tok_s    = 0.0
    t0       = time.time()
    t_log    = time.time()

    pbar = tqdm(total=args.num_iters, desc="train", unit="step", dynamic_ncols=True)

    while step < args.num_iters:
        sample = next(data_iter)
        prompt_tokens = mx.array(sample["prompt_ids"])
        ground_truth = sample["ground_truth"]

        sampler = make_sampler(temp=1.0, top_p=1.0)

        (trajectories_tokens_padded,
         trajectories_logprobs_padded,
         trajectories_masks_padded) = batched_rollout(
            policy,
            prompt_tokens,
            args.group_size,
            args.max_new_tok,
            sampler,
            tokenizer)

        rewards = gsm8k_answer_reward(
            trajectories_tokens_padded,
            trajectories_masks_padded,
            tokenizer,
            ground_truth,
        )

        advantage = (rewards - rewards.mean()) / (rewards.std() + args.epsilon)
        advantage = advantage[:, None]

        ref_logprobs = token_logprobs(
            ref,
            prompt_tokens,
            trajectories_tokens_padded,
            trajectories_masks_padded)

        def grpo_loss(model):
            new_logprobs = token_logprobs(
                model, prompt_tokens, trajectories_tokens_padded, trajectories_masks_padded)

            ratio = mx.exp(new_logprobs - trajectories_logprobs_padded)
            unclipped = ratio * advantage
            clipped = mx.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantage
            surrogate = mx.minimum(unclipped, clipped)

            kl = mx.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1
            per_token = -(surrogate - args.kl_coef * kl)

            return (per_token * trajectories_masks_padded).sum() / trajectories_masks_padded.sum()

        loss_and_grad = nn.value_and_grad(policy, grpo_loss)

        epoch_losses = []
        for _ in range(args.ppo_epochs):
            loss, grads = loss_and_grad(policy)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(policy, grads)
            epoch_losses.append(loss.item())

        ntoks = trajectories_masks_padded.sum()
        mx.eval(policy.parameters(), optimizer.state, ntoks)

        # ---- Logging -------------------------------------------------------
        loss_val  = sum(epoch_losses) / len(epoch_losses)
        lr_val    = float(optimizer.learning_rate)
        reward_val = rewards.mean().item()
        adv_mean = advantage.squeeze().mean().item()
        adv_std = advantage.squeeze().std().item()

        writer.add_scalar("train/loss", loss_val, step)
        writer.add_scalar("train/learning_rate", lr_val, step)
        writer.add_scalar("train/reward", reward_val, step)
        writer.add_scalar("train/advantage_mean", adv_mean, step)
        writer.add_scalar("train/advantage_std", adv_std, step)

        if step % args.log_every == 0:
            dt    = time.time() - t_log
            tok_s = ntoks.item() * args.log_every / max(dt, 1e-8) if step > 0 else 0.0
            writer.add_scalar("train/tokens_per_sec", tok_s, step)
            t_log = time.time()

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            lr=f"{lr_val:.2e}",
            tok_s=f"{tok_s:.0f}",
        )
        pbar.update(1)

        if step % args.param_log_every == 0:
            for name, param in tree_flatten(policy.trainable_parameters()):
                writer.add_histogram(f"params/{name}", np.array(param), step)

        # ---- Validation ----------------------------------------------------
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            val_reward = evaluate(policy, val_samples, tokenizer, args)
            writer.add_scalar("val/reward", val_reward, step)
            pbar.write(f"  [eval] step {step:5d} | val_reward {val_reward:.4f}")

        # ---- Checkpoint ----------------------------------------------------
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            pbar.write(f"  [ckpt] step {step:5d} | saving checkpoint ...")
            save_full_checkpoint(policy, step, args)

        step += 1

    pbar.close()

    if args.save_every > 0:
        save_full_checkpoint(policy, step - 1, args)

    writer.close()

    if args.debug and val_samples:
        sample = val_samples[0]
        print("=== sample after training (first debug question) ===")
        print(f"Q: {sample['question']}")
        print(f"ground truth: {sample['ground_truth']}")
        text = generate(
            policy,
            tokenizer,
            prompt=tokenizer.decode(sample["prompt_ids"]),
            max_tokens=args.max_new_tok,
            sampler=make_sampler(temp=1.0, top_p=1.0),
        )
        pred = gsm8k_grpo.extract_final_answer(text)
        print(f"model output:\n{text}")
        print(f"extracted answer: {pred}")

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print(f"TensorBoard logs:  tensorboard --logdir={args.tensorboard_dir}")


if __name__ == "__main__":
    main()
