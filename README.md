# Post-Training Lab

A minimal, hackable implementation of the full LLM post-training pipeline:
supervised fine-tuning (SFT), reinforcement learning with GRPO, evaluation, and
inference. Each stage is a self-contained script you can read top to bottom,
understand completely, and bend to your own dataset or reward.

The priority is clarity over abstraction: no framework to learn, no backend
layer hiding the algorithm, and no need to trace a dozen files to understand a
loss or training step. Implementations use native MLX on Apple Silicon and
PyTorch with CUDA on NVIDIA GPUs.

The implementations are designed to make post-training practical on consumer
hardware rather than requiring datacenter GPUs. The experiments in this
repository are run on an NVIDIA GeForce RTX 5060 Ti with 16 GB of VRAM and an
Apple M4 chip.

## Choose a backend

| Backend | Hardware | Status | Documentation |
| ------- | -------- | ------ | ------------- |
| MLX | Apple Silicon | Supported | [MLX backend](mlx_backend/README.md) |
| PyTorch (CUDA) | NVIDIA GPU | Supported | [PyTorch (CUDA) backend](cuda_backend/README.md) |

## Supported algorithms

| Algorithm | MLX (Apple Silicon) | PyTorch (CUDA) |
| --------- | ------------------- | -------------- |
| SFT | ✅ | ✅ |
| GRPO | ✅ | ✅ |

## Tutorials

Each tutorial is a self-contained implementation that can be read and changed
without navigating through the rest of the repository. See the
[tutorial instructions](tutorials/README.md) for the general run command.

| Tutorial | Backend | Code | Presentation | Video |
| -------- | ------- | ---- | ------------ | ----- |
| 👉 Minimal GRPO | PyTorch (CUDA) | [`grpo_minimal_pytorch.py`](tutorials/grpo_minimal_pytorch.py) | [View slides](tutorials/GRPO.pdf) | [Watch on YouTube](https://www.youtube.com/watch?v=vVJjUglOURs) |

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
│   ├── sft_train.py         # Supervised fine-tuning with LoRA or full training
│   ├── grpo_train.py
│   ├── generate_text.py
│   ├── gsm8k_eval.py
│   ├── numinamath_eval.py
│   ├── metamathqa_eval.py
│   └── xlam_function_calling_eval.py
├── data_preparation/
│   ├── gsm8k.py             # Shared samples, DataLoaders, and answer matching
│   ├── numinamath.py
│   ├── metamathqa.py
│   ├── xlam_function_calling.py
│   └── sft.py               # SFT padding and batching shared by all datasets
├── tutorials/
│   ├── README.md
│   └── grpo_minimal_pytorch.py # Self-contained minimal PyTorch GRPO
├── checkpoints/             # Backend-qualified checkpoints (gitignored)
└── runs/                    # Backend-qualified TensorBoard logs (gitignored)
```

Training runs use `runs/<backend>/<algorithm>/<algorithm>_<timestamp>/` with
matching checkpoints in
`checkpoints/<backend>/<algorithm>/<algorithm>_<timestamp>/`.

Generate text or evaluate a PyTorch/CUDA checkpoint on the GSM8K test split:

```bash
uv run --extra cuda -m cuda_backend.generate_text --model_path <checkpoint>
uv run --extra cuda -m cuda_backend.gsm8k_eval --model_path <checkpoint>
```

Add `--load-adapter` when `<checkpoint>` is a PEFT adapter checkpoint. See the
[CUDA backend documentation](cuda_backend/README.md) for complete examples.

Evaluate on the local NuminaMath Algebra test set with the `eval` extra:

```bash
uv run --extra cuda --extra eval -m cuda_backend.numinamath_eval --model_path <checkpoint>
```

Prepare MetaMathQA locally and evaluate its test split:

```bash
uv run --extra eval python data/metamathqa/prepare.py
uv run --extra cuda --extra eval -m cuda_backend.metamathqa_eval --model_path <checkpoint>
```

Preparation deduplicates exact `query` matches by default and removes
unparseable reference answers before splitting. Use
`--keep-duplicate-queries` to retain repeated queries. See the
[MetaMathQA data README](data/metamathqa/README.md) for split settings and evaluation options.

Prepare xLAM function-calling data and evaluate its local test split:

```bash
uv run python data/xlam-function-calling-60k/prepare.py
uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 100
```

The evaluator uses each model's native tool-aware chat template and reports
exact call-set and function-name-set accuracy. See the
[xLAM data README](data/xlam-function-calling-60k/README.md) for preparation,
grading, SFT, and reproducibility details.

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
