# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/). Pre-1.0 releases are early and the
API may change between minor versions.

## [0.1.0] — 2026-07-01

First public release.

### Added
- SFT training script ([`sft_train.py`](mlx_backend/sft_train.py)) — LoRA fine-tuning
  on GSM8K (chain-of-thought).
- GRPO training script ([`grpo_train.py`](mlx_backend/grpo_train.py)) — group-relative
  policy optimization with a clipped surrogate objective and KL penalty against a
  frozen reference, with a `--debug` overfit sanity-check mode.
- SFT → GRPO resume via `--load-adapter` (continue LoRA training from SFT
  adapters; reference model becomes the SFT policy).
- Text generation script ([`generate_text.py`](mlx_backend/generate_text.py)) for
  base models or fused/adapter checkpoints.
- Data preparation for GSM8K ([`data_preparation/`](data_preparation/)).
- TensorBoard logging and fused checkpoints loadable by `mlx_lm.load()`.
- MLX / Apple Silicon backend. (A CUDA backend is planned.)
