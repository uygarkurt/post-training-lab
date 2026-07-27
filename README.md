# Post-Training Lab

A minimal, hackable implementation of LLM post-training. Each training stage is
a self-contained script you can read top to bottom, understand completely, and
bend to your own dataset or reward.

The priority is clarity over abstraction: no framework to learn, no backend
layer hiding the algorithm, and no need to trace a dozen files to understand a
loss or training step.

## Choose a backend

| Backend | Hardware | Status | Documentation |
| ------- | -------- | ------ | ------------- |
| MLX | Apple Silicon | Supported | [MLX backend](mlx_backend/README.md) |
| CUDA | NVIDIA GPU | In progress | [CUDA backend](cuda_backend/README.md) |

## Supported algorithms

| Algorithm | MLX (Apple Silicon) | CUDA (NVIDIA GPU) |
| --------- | ------------------- | ----------------- |
| SFT | ✅ | ❌ |
| GRPO | ✅ | ❌ |

## Shared requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

This repository uses one project environment. Install and run only the optional
backend dependencies you need. The currently runnable MLX commands use
`--extra mlx`; CUDA dependencies will be declared when that implementation is
available.

## Project layout

```text
post-training-lab/
├── mlx_backend/             # Runnable Apple Silicon implementation
│   ├── sft_train.py
│   ├── grpo_train.py
│   └── generate_text.py
├── cuda_backend/            # CUDA status and future implementation
├── data_preparation/
│   └── gsm8k.py             # Shared samples, DataLoaders, and answer matching
├── checkpoints/             # Backend-qualified checkpoints (gitignored)
└── runs/                    # Backend-qualified TensorBoard logs (gitignored)
```

## Citation

If you use this software, please cite it:

```bibtex
@software{uygarkurt_2026_21111797,
  author       = {uygarkurt},
  title        = {uygarkurt/post-training-lab: v0.1.0},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.21111797},
  url          = {https://doi.org/10.5281/zenodo.21111797},
  swhid        = {swh:1:dir:347ae7d4023d5d6c28e84943b71de9b65e2aa721
                   ;origin=https://doi.org/10.5281/zenodo.21111796;vi
                   sit=swh:1:snp:4b6347ae5b75116e19830f6ac9d34825545c
                   7235;anchor=swh:1:rel:fc0e838f97148d60ebcc320be26c
                   f0a38a248686;path=uygarkurt-post-training-
                   lab-449dadb
                  },
}
```

## License

[MIT](LICENSE)
