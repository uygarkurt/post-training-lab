# Tutorials

The tutorials are self-contained implementations that keep data preparation,
training, evaluation, and checkpoint saving in one readable Python file.

## Available tutorials

| Tutorial | Code | Presentation | Video |
| -------- | ---- | ------------ | ----- |
| Minimal GRPO in PyTorch | [`grpo_minimal_pytorch.py`](grpo_minimal_pytorch.py) | [View slides](GRPO.pdf) | [Watch on YouTube](https://www.youtube.com/watch?v=vVJjUglOURs) |

## Setup

Tutorials require Python 3.12 or newer, [`uv`](https://docs.astral.sh/uv/),
and hardware supported by the selected backend. Install the dependencies from
the repository root:

```bash
uv sync --extra <backend>
```

Use `cuda` for NVIDIA GPUs or `mlx` for Apple Silicon.

## Run

Run any tutorial from the repository root:

```bash
uv run --extra <backend> python tutorials/<tutorial>.py
```

Tutorial settings are uppercase constants near the beginning of each script,
so they can be changed without command-line arguments.
