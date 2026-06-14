import torch
import torch.nn.functional as F
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import mlx.optimizers as optim
import mlx.core as mx

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
GROUP_SIZE   = 8       # G: completions sampled per prompt
MAX_NEW_TOK  = 20      # completion length
LR           = 1e-4    # advantages are normalized to ~O(1), so a tiny LR barely
                       # moves the policy. ~1e-4 makes the reward actually drive it.
KL_COEF      = 0.02    # beta: strength of the KL-to-reference penalty. Too high and
                       # the anchor to the reference overpowers the reward signal.
CLIP_EPS     = 0.2     # PPO clip range
PPO_EPOCHS   = 4       # gradient steps taken per rollout (reuses the same group)
STEPS        = 100
TARGET_WORD  = " the"  # dummy reward target; model should learn to spam this.
                       # Pick a COMMON token: GRPO learns from reward *variance*
                       # within a group, so the reward must be nonzero often
                       # enough that groups differ. A rare target (e.g. " good")
                       # gives all-zero groups -> zero advantage -> no learning.

# A few fixed prompts. GRPO doesn't need labels, just prompts to roll out from.
PROMPTS = [
    "I think that",
    "The weather today is",
    "My favorite thing about life is",
    "Once upon a time",
]

def main():
    policy, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target_id = tokenizer.encode(TARGET_WORD)[0]

    ref, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")

    optimizer = optim.AdamW(learning_rate=LR)

    for step in range(STEPS):
        prompt = PROMPTS[step % len(PROMPTS)]
        input_ids = mx.array([tokenizer.encode(prompt)]) # [B, SEQ_LEN]
        prompt_len = input_ids.shape[1]

        prompt_tokens = mx.array(tokenizer.encode(prompt))  # 1D: (seq_len,)

        trajectories = []
        for _ in range(GROUP_SIZE):
            tokens = []
            logprobs = []
            for response in stream_generate(policy, tokenizer, prompt=prompt_tokens, max_tokens=MAX_NEW_TOK, sampler=make_sampler(temp=1.0, top_p=1.0)):
                tokens.append(response.token)
                logprobs.append(response.logprobs[response.token])
                if response.finish_reason is not None:
                    break
            trajectories.append({"tokens": tokens, "logprobs": logprobs})


if __name__ == "__main__":
    main()