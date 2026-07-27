# Post-Training Lab — Agent Guidance

## Project purpose

This repository is a minimal, hackable implementation of the full LLM
post-training pipeline. It is primarily an educational and experimentation
codebase.

## Design priorities

When making changes, prioritize:

1. Understandability
2. Minimal implementation
3. Correctness
4. Ease of experimentation
5. Performance and extensibility

Prefer code that can be read from top to bottom without navigating through
multiple abstraction layers.

## Implementation rules

- Preserve the self-contained nature of the training scripts.
- Prefer explicit code over framework-like abstractions.
- Avoid introducing classes, registries, factories, configuration frameworks,
  or extra modules unless they remove substantial unavoidable complexity.
- Do not deduplicate code when doing so would make an individual algorithm
  harder to understand in isolation.
- Keep the mathematical and algorithmic steps of every post-training method
  visible in its training script.
- Use descriptive variable names, especially for losses, rewards, advantages,
  masks, token probabilities, and other algorithm-specific quantities.
- Keep new dependencies to a minimum.
- Make the smallest change that satisfies the request.
- Preserve hackability: changing an algorithm, loss, reward, dataset, or model
  backend should remain straightforward.

## Repository conventions

- Use Python 3.12 or newer.
- Use `uv` for dependency management and command execution.
- Support both MLX and CUDA backends. Do not assume that their implementations
  have complete feature parity; preserve the simplicity and native conventions
  of each backend.
- Avoid backend abstractions that obscure an algorithm merely to share code
  between MLX and CUDA.
- Do not modify generated checkpoints or TensorBoard runs.

## Verification

- Run focused, inexpensive checks where possible.
- Do not start long-running model training unless the user explicitly asks.
- Prefer debug or smoke-test configurations when runtime validation is needed.
- State clearly when backend-specific execution could not be tested on the
  current hardware.
