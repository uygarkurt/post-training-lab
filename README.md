# Post-Training Lab

A minimal, hackable implementation of the full LLM post-training pipeline:
supervised fine-tuning (SFT), reinforcement learning with GRPO, evaluation, and
inference. Each stage is a self-contained script you can read top to bottom,
understand completely, and bend to your own dataset or reward.

The priority is clarity over abstraction: no framework to learn, no backend
layer hiding the algorithm, and no need to trace a dozen files to understand a
loss or training step. Implementations use native MLX on Apple Silicon and
PyTorch with CUDA on NVIDIA GPUs.

## Choose a backend

| Backend | Hardware | Status | Documentation |
| ------- | -------- | ------ | ------------- |
| MLX | Apple Silicon | Supported | [MLX backend](mlx_backend/README.md) |
| PyTorch (CUDA) | NVIDIA GPU | In progress | [PyTorch (CUDA) backend](cuda_backend/README.md) |

## Supported algorithms

| Algorithm | MLX (Apple Silicon) | PyTorch (CUDA) |
| --------- | ------------------- | -------------- |
| SFT | ✅ | ❌ |
| GRPO | ✅ | ✅ |

## Shared requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

This repository uses one project environment. Install and run only the optional
backend dependencies you need. Use `--extra mlx` for MLX commands and
`--extra cuda` for CUDA commands.

## Project layout

```text
post-training-lab/
├── mlx_backend/             # Runnable Apple Silicon implementation
│   ├── sft_train.py
│   ├── grpo_train.py
│   ├── generate_text.py
│   └── gsm8k_eval.py
├── cuda_backend/            # Runnable PyTorch implementation
│   ├── grpo_train.py
│   ├── generate_text.py
│   └── gsm8k_eval.py
├── data_preparation/
│   └── gsm8k.py             # Shared samples, DataLoaders, and answer matching
├── checkpoints/             # Backend-qualified checkpoints (gitignored)
└── runs/                    # Backend-qualified TensorBoard logs (gitignored)
```

PyTorch/CUDA GRPO runs use `runs/cuda/grpo_<timestamp>/` with matching
checkpoints in `checkpoints/cuda/grpo_<timestamp>/`.

Generate text or evaluate a PyTorch/CUDA checkpoint on the GSM8K test split:

```bash
uv run --extra cuda -m cuda_backend.generate_text --model_path <checkpoint>
uv run --extra cuda -m cuda_backend.gsm8k_eval --model_path <checkpoint>
```

Add `--load-adapter` when `<checkpoint>` is a PEFT adapter checkpoint. See the
[CUDA backend documentation](cuda_backend/README.md) for complete examples.

## Citation

If you use this software, please cite it using the concept DOI below, which
represents all versions and resolves to the latest release:

```bibtex
@software{kurt_post_training_lab,
  author    = {Kurt, Uygar},
  title     = {Post-Training Lab},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21111796},
  url       = {https://doi.org/10.5281/zenodo.21111796}
}
```

## License

[MIT](LICENSE)
