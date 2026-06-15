from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import mlx.optimizers as optim
import mlx.core as mx

GROUP_SIZE   = 8
MAX_NEW_TOK  = 20
LR           = 1e-4
KL_COEF      = 0.02
CLIP_EPS     = 0.2
PPO_EPOCHS   = 4 
STEPS        = 100
TARGET_WORD  = " the" 

PROMPTS = [
    "I think that",
    "The weather today is",
    "My favorite thing about life is",
    "Once upon a time",
]

def word_repetition_reward(trajectory_tokens_padded, target_id):
    is_target = trajectory_tokens_padded == target_id
    return is_target.sum(axis=-1)  # [G]

def pad_trajectories(
    trajectories_tokens, 
    trajectories_logprobs, 
    tokenizer):
    max_len = max(len(t) for t in trajectories_tokens)
    trajectories_tokens_padded = []
    trajectories_logprobs_padded = []
    trajectories_masks_padded = []
    for tokens, logprobs in zip(trajectories_tokens, trajectories_logprobs):
        pad_len = max_len - len(tokens)
        trajectories_tokens_padded.append(tokens + [tokenizer.pad_token_id] * pad_len)
        trajectories_logprobs_padded.append(logprobs + [0.0] * pad_len)
        trajectories_masks_padded.append([1] * len(tokens) + [0] * pad_len)

    trajectories_tokens_padded = mx.array(trajectories_tokens_padded) # [G, L]
    trajectories_logprobs_padded = mx.array(trajectories_logprobs_padded) # [G, L]
    trajectories_masks_padded = mx.array(trajectories_masks_padded) # [G, L]

    return (trajectories_tokens_padded,
            trajectories_logprobs_padded,
            trajectories_masks_padded)

def main():
    policy, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target_id = tokenizer.encode(TARGET_WORD)[0]

    ref, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")

    optimizer = optim.AdamW(learning_rate=LR)

    for step in range(STEPS):
        prompt = PROMPTS[step % len(PROMPTS)]
        prompt_tokens = mx.array(tokenizer.encode(prompt))  # [T]

        trajectories_tokens = []
        trajectories_logprobs = []
        sampler = make_sampler(temp=1.0, top_p=1.0)
        for _ in range(GROUP_SIZE):
            tokens = []
            logprobs = []
            for response in stream_generate(policy, tokenizer, prompt=prompt_tokens, max_tokens=MAX_NEW_TOK, sampler=sampler):
                tokens.append(response.token)
                logprobs.append(response.logprobs[response.token])
                if response.finish_reason is not None:
                    break
            trajectories_tokens.append(tokens)
            trajectories_logprobs.append(logprobs)

        (trajectories_tokens_padded, 
        trajectories_logprobs_padded, 
        trajectories_masks_padded) = pad_trajectories(
            trajectories_tokens,
            trajectories_logprobs,
            tokenizer)
 
        rewards = word_repetition_reward(trajectories_tokens_padded, target_id)

if __name__ == "__main__":
    main()