import argparse
import itertools
import os

from mlx_lm import load, generate
from mlx_lm.utils import save as save_model_checkpoint
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn
from tensorboardX import SummaryWriter

import data_preperation.gsm8k_grpo as gsm8k_grpo

DEBUG_PROMPTS = [
    "I think that",
    "The weather today is",
    "My favorite thing about life is",
    "Once upon a time",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training on MLX (Apple Silicon).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--debug", action="store_true", help="Use toy prompts and word-repetition reward")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="HuggingFace model name or path")

    parser.add_argument("--group-size", type=int, default=8, help="Number of rollouts per prompt (G)")
    parser.add_argument("--max-new-tok", type=int, default=256, help="Max tokens to generate per rollout")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO inner epochs per step")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--target-word", type=str, default=" the", help="Target token for debug reward")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Advantage normalisation epsilon")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffle")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction held out from GSM8K train set")
    parser.add_argument("--max-prompt-len", type=int, default=512, help="Skip GSM8K prompts longer than this")

    parser.add_argument("--tensorboard-dir", type=str, default="./runs/grpo", help="Directory for TensorBoard logs")

    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/grpo", help="Directory for checkpoints")

    args = parser.parse_args()
    if args.debug:
        args.max_new_tok = 20
    return args


def word_repetition_reward(trajectory_tokens_padded, target_id):
    is_target = trajectory_tokens_padded == target_id
    return is_target.sum(axis=-1)  # [G]


def gsm8k_answer_reward(trajectory_tokens, masks, tokenizer, ground_truth):
    rewards = []
    for g in range(trajectory_tokens.shape[0]):
        valid_ids = [
            int(t) for t, m in zip(trajectory_tokens[g].tolist(), masks[g].tolist())
            if m == 1
        ]
        text = tokenizer.decode(valid_ids)
        pred = gsm8k_grpo.extract_gsm8k_answer(text)
        rewards.append(1.0 if pred == ground_truth else 0.0)
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


def save_checkpoint(model, tokenizer, config, step, args):
    """Save a full checkpoint loadable by mlx_lm.load()."""
    out_dir = os.path.join(args.checkpoint_dir, f"step_{step:06d}")
    os.makedirs(out_dir, exist_ok=True)
    mx.eval(model.parameters())
    save_model_checkpoint(out_dir, args.model, model, tokenizer, config, donate_model=False)
    print(f"  checkpoint -> {out_dir}/  (load with: mlx_lm.load('{out_dir}'))")


def main():
    args = parse_args()
    mode = "debug" if args.debug else "gsm8k"
    print(f"Mode: {mode} | max_new_tok={args.max_new_tok} | steps={args.steps}")

    policy, tokenizer, config = load(args.model, return_config=True)
    policy.set_dtype(mx.float32)
    policy.train()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ref, _ = load(args.model)
    ref.set_dtype(mx.float32)

    optimizer = optim.AdamW(learning_rate=args.lr)

    if args.debug:
        target_id = tokenizer.encode(args.target_word)[0]
        sample_prompt = DEBUG_PROMPTS[0]
        data_iter = None
    else:
        print("Loading GSM8K dataset ...")
        gsm8k_samples = gsm8k_grpo.build_grpo_samples(tokenizer, args)
        data_iter = itertools.cycle(gsm8k_samples)

    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(f"TensorBoard logs -> {args.tensorboard_dir}  (run: tensorboard --logdir={args.tensorboard_dir})")

    reward_history = []
    for step in range(args.steps):
        if args.debug:
            prompt = DEBUG_PROMPTS[step % len(DEBUG_PROMPTS)]
            prompt_tokens = mx.array(tokenizer.encode(prompt))
            ground_truth = None
        else:
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

        if args.debug:
            rewards = word_repetition_reward(trajectories_tokens_padded, target_id)
        else:
            rewards = gsm8k_answer_reward(
                trajectories_tokens_padded,
                trajectories_masks_padded,
                tokenizer,
                ground_truth,
            )

        advantage = (rewards - rewards.mean()) / (rewards.std() + args.epsilon) # [G]
        advantage = advantage[:, None] # [G, 1]

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
            mx.eval(policy.parameters(), optimizer.state)
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        reward_val = rewards.mean().item()
        reward_history.append(reward_val)
        moving_avg = sum(reward_history[-10:]) / len(reward_history[-10:])
        adv_mean = advantage.squeeze().mean().item()
        adv_std = advantage.squeeze().std().item()

        writer.add_scalar("train/loss", avg_loss, step)
        writer.add_scalar("train/reward", reward_val, step)
        writer.add_scalar("train/reward_avg10", moving_avg, step)
        writer.add_scalar("train/advantage_mean", adv_mean, step)
        writer.add_scalar("train/advantage_std", adv_std, step)

        print(f"step {step:3d} | reward {reward_val:5.2f} | avg10 {moving_avg:5.2f} | "
              f"loss {avg_loss:+.4f} | adv {adv_mean:+.3f}")

        if args.save_steps > 0 and step > 0 and step % args.save_steps == 0:
            print(f"  [ckpt] step {step:5d} | saving checkpoint ...")
            save_checkpoint(policy, tokenizer, config, step, args)

    writer.close()

    if args.save_steps > 0:
        save_checkpoint(policy, tokenizer, config, args.steps - 1, args)

    if args.debug:
        print("=== sample after training ===")
        text = generate(
            policy,
            tokenizer,
            prompt=sample_prompt,
            max_tokens=args.max_new_tok,
            sampler=make_sampler(temp=1.0, top_p=1.0),
        )
        print(text)


if __name__ == "__main__":
    main()
