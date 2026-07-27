# CUDA Backend

CUDA support for NVIDIA GPUs is in progress and is not runnable yet.

The implementation will follow the same visible entrypoint structure as MLX:

- `sft_train.py`
- `grpo_train.py`
- `generate_text.py`

CUDA-specific dependencies and installation commands will be added when the
first runnable implementation lands. Until then, the root project does not
declare a `cuda` dependency extra or provide placeholder scripts.

CUDA entrypoints will consume the same backend-neutral samples and answer
matching functions from `data_preparation/gsm8k.py`, including its PyTorch
DataLoaders. CUDA device placement, models, losses, and optimization steps will
remain in the CUDA scripts.
