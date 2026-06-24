from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

GROUP_SIZE   = 8
MAX_NEW_TOK  = 20 # L
LR           = 1e-6
KL_COEF      = 0.02
CLIP_EPS     = 0.2
PPO_EPOCHS   = 4 
STEPS        = 100
TARGET_WORD  = " the" 
EPSILON      = 1e-8

PROMPTS = [
    "I think that",
    "The weather today is",
    "My favorite thing about life is",
    "Once upon a time",
]

def word_repetition_reward(trajectory_tokens_padded, target_id):
    is_target = trajectory_tokens_padded == target_id
    return is_target.sum(axis=-1)  # [G]

def batched_rollout(model, prompt_tokens, group_size, max_new_tok, sampler, tokenizer):
    prompt_batch = mx.repeat(prompt_tokens[None, :], group_size, axis=0) # [G, P]

    cache = make_prompt_cache(model)
    logits = model(prompt_batch, cache=cache)[:, -1, :] # [G, V]

    eos_id = tokenizer.eos_token_id
    # `active[g]` is True while sequence g has not yet emitted EOS.
    active = mx.ones((group_size,), dtype=mx.bool_)
   
    # We collect one [G] array per decode step, then stack into [G, L] at the end.
    tokens_steps = []
    logprobs_steps = []
    masks_steps = []
    for _ in range(max_new_tok):
        # Turn the current logits [G, V] into log-probabilities [G, V].
        lp = nn.log_softmax(logits, axis=-1)

        # Sample the next token for every sequence at once -> [G].
        next_tok = sampler(lp)  # [G]

        # Look up the log-prob the policy assigned to the token it sampled
        tok_logprob = lp[mx.arange(group_size), next_tok]   # [G]

        # token, so an EOS token is itself valid (mask=1); tokens generated
        # after a sequence's EOS get masked out after the loop.
        masks_steps.append(active.astype(mx.int32))   
        tokens_steps.append(next_tok)
        logprobs_steps.append(tok_logprob)

        # 7) Update `active`: a sequence stays active only if it did NOT just
        #    emit EOS. Hint: combine the old `active` with (next_tok != eos_id).
        active = active & (next_tok != eos_id) #[G] bool

        logits = model(next_tok[:, None], cache=cache)[:, -1, :]  # [G, V]

    # Stack the per-step [G] arrays into [G, L].
    tokens = mx.stack(tokens_steps, axis=-1)    # [G, L]
    logprobs = mx.stack(logprobs_steps, axis=-1)  # [G, L]
    masks = mx.stack(masks_steps, axis=-1)     # [G, L]

 
    # Replace masked-out (post-EOS) positions with the pad token, and zero out their log-probs so they don't affect loss.
    tokens = mx.where(masks == 1, tokens, tokenizer.pad_token_id) # [G, L] with padding in masked positions
    logprobs = logprobs * masks # [G, L] with zeros in masked positions
    return tokens, logprobs, masks

def token_logprobs(model, prompt_tokens, input_ids, loss_mask):
    G = input_ids.shape[0]
    P = prompt_tokens.shape[0]

    # prompt_tokens[None, :] -> [1, P], unsqueeze(0)
    prompt_batch = mx.repeat(prompt_tokens[None, :], G, axis=0) # [G, P]
    full_ids = mx.concatenate([prompt_batch, input_ids], axis=-1) # [G, L+P]

    logits = model(full_ids) # [G, L+P, V]
    logprobs = nn.log_softmax(logits, axis=-1) # [G, L+P, V]

    # logprobs[:, i, :] is the prediction for the token at position i+1.
    # The last prompt token (row P-1) predicts the 1st generated token, so we
    # keep its score. We drop the final row since it predicts a token past the
    # end of the sequence, which has no target.
    gen_logprobs = logprobs[:, P-1:-1, :] # [G, L, V]

    g_ids = mx.expand_dims(mx.arange(G), axis=-1) # [G, 1]
    l_idx = mx.expand_dims(mx.arange(input_ids.shape[1]), 0) # [1, L]
    final_logprobs = gen_logprobs[g_ids, l_idx, input_ids] # [G, L]

    return final_logprobs * loss_mask

def main():
    policy, tokenizer = load("Qwen/Qwen2-0.5B-Instruct")
    policy.set_dtype(mx.float32)
    policy.train()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target_id = tokenizer.encode(TARGET_WORD)[0]

    ref, _ = load("Qwen/Qwen2-0.5B-Instruct")
    ref.set_dtype(mx.float32)

    optimizer = optim.AdamW(learning_rate=LR)

    reward_history = []
    for step in range(STEPS):
        prompt = PROMPTS[step % len(PROMPTS)]
        prompt_tokens = mx.array(tokenizer.encode(prompt))  # [P]

        sampler = make_sampler(temp=1.0, top_p=1.0)

        # [G, L]
        (trajectories_tokens_padded,
        trajectories_logprobs_padded,
        trajectories_masks_padded) = batched_rollout(
            policy,
            prompt_tokens,
            GROUP_SIZE,
            MAX_NEW_TOK,
            sampler,
            tokenizer)

        rewards = word_repetition_reward(trajectories_tokens_padded, target_id)

        advantage = (rewards - rewards.mean()) / (rewards.std() + EPSILON) # [G]
        advantage = advantage[:, None] # [G, 1]

        ref_logprobs = token_logprobs(
            ref,
            prompt_tokens,
            trajectories_tokens_padded,
            trajectories_masks_padded)

        # The policy forward must run inside the differentiated function so
        # value_and_grad can trace grads w.r.t. its params (MLX has no
        # loss.backward()). Per-step constants are captured from the closure.
        def grpo_loss(model):
            new_logprobs = token_logprobs(
                model, prompt_tokens, trajectories_tokens_padded, trajectories_masks_padded)

            ratio = mx.exp(new_logprobs - trajectories_logprobs_padded)
            unclipped = ratio * advantage
            clipped = mx.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
            surrogate = mx.minimum(unclipped, clipped)

            kl = mx.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1
            per_token = -(surrogate - KL_COEF * kl)

            # Token-level mean over valid tokens keeps the gradient scale
            # invariant to sequence length, so a fixed LR stays consistent.
            return (per_token * trajectories_masks_padded).sum() / trajectories_masks_padded.sum()

        loss_and_grad = nn.value_and_grad(policy, grpo_loss)

        epoch_losses = []
        for _ in range(PPO_EPOCHS):
            loss, grads = loss_and_grad(policy)
            # Clip the global grad norm: a single unconstrained update can move a
            # token's logprob enough to blow up the k3 KL term (exp(ref - new)),
            # which destabilizes training. Standard PPO/GRPO practice.
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)
            epoch_losses.append(loss.item())

        # One line per step. Mean reward is the real signal that the policy is
        # learning; the PPO loss is averaged over the inner epochs for context.
        # Per-step reward is noisy, so track a moving average to see the trend.
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        reward_history.append(rewards.mean().item())
        moving_avg = sum(reward_history[-10:]) / len(reward_history[-10:])
        print(f"step {step:3d} | reward {reward_history[-1]:5.2f} | avg10 {moving_avg:5.2f} | loss {avg_loss:+.4f}")

    print("=== sample after training ===")
    text = generate(
        policy,
        tokenizer,
        prompt=PROMPTS[0],
        max_tokens=MAX_NEW_TOK,
        sampler=make_sampler(temp=1.0, top_p=1.0),
    )
    print(text)

if __name__ == "__main__":
    main()