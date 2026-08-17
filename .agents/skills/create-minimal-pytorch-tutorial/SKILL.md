---
name: create-minimal-pytorch-tutorial
description: Create self-contained minimal PyTorch tutorial implementations by extracting canonical code from this repository without rewriting algorithm logic. Use when asked to turn an existing CUDA/PyTorch training algorithm such as GRPO, SFT, or DPO into a flat tutorial Python file and accompanying documentation while preserving reused functions, comments, tensor-shape annotations, validation, deterministic seeding, terminal output, and checkpoint behavior.
---

# Create a Minimal PyTorch Tutorial

Create a readable, single-file tutorial from the repository's current canonical implementation. Treat this as code extraction and assembly, not a rewrite.

## Inspect the Current Implementation

1. Read the applicable `AGENTS.md` instructions and inspect the worktree before editing.
2. Locate the canonical PyTorch/CUDA training entrypoint and every helper it calls for data preparation, rewards, evaluation, seeding, and saving.
3. Read all relevant source sections completely. Rediscover them on every invocation; never assume an earlier tutorial still matches the canonical code.
4. Identify algorithm-critical calculations separately from experiment-management features.
5. Ask only when multiple canonical variants exist and the user's intended variant cannot be inferred safely.

## Preserve the Canonical Code

- Copy required function bodies, calculations, variable names, docstrings, comments, and tensor-shape comments verbatim.
- Do not refactor, simplify, optimize, or replace copied algorithm logic with an alternative implementation.
- Permit only these mechanical integration edits:
  - Replace command-line `args.*` accesses with descriptive uppercase constants.
  - Replace cross-module qualifiers with same-file references.
  - Add the smallest orchestration glue needed to connect extracted sections and label results.
  - Remove calculations whose only consumer was an intentionally removed logging system; never remove values used by the algorithm.
- Copy only the branches needed by the tutorial. Exclude unrelated debug modes, resume paths, alternate training modes, and backend variants.
- Keep imports sorted and formatted. Do not leave unused imports, variables, or metric-only calculations.

## Build the Tutorial

1. Use the flat path `tutorials/<algorithm>_minimal_pytorch.py`; do not add an algorithm subdirectory unless the user requests one.
2. Put all hyperparameters, model names, dataset settings, evaluation cadence, save cadence, and output paths near the top as uppercase constants. Do not add a command-line parser.
3. Inline only the relevant canonical pieces:
   - Dataset loading, tokenization, splitting, collation, and batching
   - Reward or scoring functions
   - Deterministic Python, NumPy, PyTorch, and CUDA seeding
   - Policy, reference model, adapter, optimizer, and scheduler setup required by the algorithm
   - The complete training objective and update loop
   - Validation and test evaluation
   - Checkpoint and tokenizer saving
4. Preserve canonical terminal behavior. Omit TensorBoard, timestamped run management, tee processes, argument dumps, and persistent metrics unless the user explicitly requests them.
5. Evaluate the official test set once before any optimizer update and once after training using the canonical evaluation function. Label both results clearly, and never use test results for optimization, early stopping, or checkpoint selection.
6. Retain canonical validation during training. Do not invent a validation loss when the implementation uses an accuracy or task-specific metric.
7. Save generated weights outside the source tree under `checkpoints/tutorials/<algorithm>_minimal_pytorch/step_XXXXXX`. Preserve canonical periodic and final-save behavior.
8. Create a feature branch named `tutorials/<algorithm>-minimal-pytorch` only when the user requests branch creation.

## Document It

- Create or update `tutorials/README.md` with the tutorial title, purpose, hardware and dependency requirements, exact `uv` command, constants-based configuration, evaluation behavior, terminal output, checkpoint location, reproducibility limits, and link to the canonical implementation.
- Add a concise discoverability link and layout entry to the root README.
- State that the tutorial is assembled from canonical code and name the source files.
- If other tutorials already exist, extend their catalog without overwriting their documentation.

## Verify Fidelity and Quality

1. Run Ruff on the tutorial and fix import order, formatting diagnostics, unused imports, and unused variables without changing algorithm behavior.
2. Compile and import the script without invoking training.
3. Exercise reward or scoring helpers with representative correct, incorrect, formatted-number, fallback, and missing-answer cases when applicable.
4. Compare copied helpers against their canonical sources with text or AST checks, allowing only the approved mechanical substitutions.
5. Confirm statically that:
   - There is no argument parser, TensorBoard use, or repository-local import.
   - The test dataset is evaluated exactly before and after training and is never used for training decisions.
   - Validation and checkpoint cadence match the constants.
   - Final-save behavior handles a last step that is not on the periodic boundary.
6. Run `git diff --check` and remove generated caches from `tutorials/`.
7. Do not start full training unless the user explicitly requests it. Report when CUDA runtime verification is unavailable.

## Report the Result

Lead with the branch and created tutorial. Summarize documentation and output locations, list focused checks that passed, state CUDA limitations, and say whether changes remain uncommitted.
