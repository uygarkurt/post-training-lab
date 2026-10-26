---
name: convert-cuda-to-mlx
description: Convert this repository's canonical CUDA/PyTorch algorithms into MLX implementations, replacing existing MLX counterparts. Use when asked to port or mirror a CUDA algorithm to MLX while preserving its CLI arguments, literal defaults, variable names, mathematical steps, and overall flow as closely as possible.
---

# Convert CUDA Algorithms to MLX

Use the current `cuda_backend/` implementation as the source of truth. Write the
matching `mlx_backend/` implementation so the files can be read side by side.
Prioritize minimality, hackability, and understandability. This is a backend
translation, not an algorithm redesign or an opportunity to preserve an older
MLX interface.

## Inspect the Source

1. Read applicable `AGENTS.md` instructions and inspect the worktree. Identify
   the requested CUDA entrypoint and its matching MLX path; retain the filename.
   If the requested algorithm has several plausible CUDA variants and the
   intended source is unclear, ask before choosing.
2. Read the complete CUDA file and the data, reward, evaluation, model-loading,
   and checkpoint helpers it uses. Rediscover the current code on each use;
   do not derive the conversion from a tutorial or an older MLX implementation.
3. Inspect the existing MLX counterpart and relevant native utilities for
   reusable backend operations. CUDA determines the behavior and structure.
4. Before editing, state the source and destination, the intended checks, and
   any known differences that need a native MLX implementation.

## Preserve the Command Contract

- Copy the CUDA argument definitions, grouping, order, and validation. Preserve
  flag spellings, aliases, destinations, types, choices, actions, requiredness,
  literal defaults, and disable semantics. Change backend labels in help text
  only where needed for accuracy.
- Preserve **all** defaults, including model IDs, dataset choices, seeds,
  hyperparameters, and output paths. Even a default containing `cuda` stays
  identical unless the user explicitly requests a different default. Do not
  substitute an MLX model ID, smaller batch, shorter sequence, or quantized model.
- A CUDA command must retain the same algorithm arguments and values after
  changing only the backend entrypoint and, if needed, the dependency extra:

  ```text
  uv run --extra cuda -m cuda_backend.sft_train [arguments]
  uv run --extra mlx -m mlx_backend.sft_train [the same arguments]
  ```

- Implement the behavior behind every option. Do not accept ignored flags,
  drop CUDA options, retain obsolete MLX-only options, or add separate MLX
  controls just to work around implementation differences.
- Check whether the same model IDs and input artifacts can actually load.
  Matching flag syntax does not prove model or adapter compatibility. If a
  library limitation prevents the requested behavior, identify the concrete
  blocker and ask about the smallest necessary deviation. Do not silently
  change semantics or claim complete parity.

## Translate in Place

- Create or replace `mlx_backend/<matching_filename>.py`. Invoking this skill
  authorizes replacement of the existing counterpart; do not ask separately
  whether to overwrite it or create a parallel legacy/versioned file. Preserve
  unrelated user edits; if they overlap the replacement and their intent is
  unclear, ask about that conflict before discarding them.
- Keep imports, constants, functions, sections, branches, loops, and execution
  order aligned with CUDA wherever possible. Preserve function names and
  signatures, algorithm variable names, comments, docstrings, and tensor-shape
  notes wherever they remain accurate. Update backend-specific explanations.
- Translate corresponding tensor operations line by line where practical.
  Use native MLX calls where execution models differ. Extract only the smallest
  loss region needed for MLX differentiation, leaving the mathematical steps
  explicit and in their original order. Do not force superficial line equality
  through wrappers or move the algorithm into a trainer framework.
- Keep losses, rewards, advantages, and update steps visible in the script.
  Do not refactor either backend to share algorithm code or add backend
  dispatchers, registries, configuration layers, or speculative features.
- Reuse existing library functions for supporting work. Check local code and
  installed dependencies first, and verify uncertain MLX APIs against official
  documentation. Add custom glue only for a concrete missing behavior and
  briefly explain why it is needed. Give new functions concise docstrings;
  do not add function signature type annotations.

## Preserve Behavior Across Backend Differences

- Reuse `data_preparation/` datasets, tokenization, splits, padding, masks,
  collators, and PyTorch DataLoaders without changing their shared contract.
  Convert batches to MLX arrays inside `mlx_backend/`. Keep MLX imports and
  device/model operations out of shared data code.
- Preserve token shifts, mask alignment, reduction axes, accumulation dtype,
  loss denominators, clipping, epsilon placement, and normalization details.
  Check library defaults such as population versus sample standard deviation;
  similarly named operations need not have identical semantics.
- Preserve gradient boundaries: frozen reference models, detached rollout or
  old-policy values, trainable parameter selection, and the exact number and
  order of optimizer updates. Use native MLX differentiation and explicitly
  evaluate lazy arrays/state where required for updates, timing, and logging.
- Match train/eval mode, dtype, optimizer settings, weight decay, scheduling,
  warmup boundaries, and the learning rate used at each update. Preserve LoRA
  targets, rank, alpha-to-scale meaning, dropout, optional adapter loading and
  merging, and full-training behavior when present in CUDA.
- Preserve rollout sampling settings, prompt grouping, padding direction,
  EOS handling, response masks, truncation, and generation limits. Seed the
  relevant Python, NumPy, PyTorch data-loading, and MLX random generators;
  do not promise identical random streams across backends.
- Match initial and periodic validation, metric weighting, validation mode
  restoration, logging cadence, TensorBoard tags, progress bars, terminal
  messages, timestamp conventions, `args.json`, and `out.log` capture.
- Match checkpoint directory construction, step numbering, periodic saves,
  final saves, and save-disable behavior. Use native serialization with the
  model/tokenizer/adapter metadata needed by MLX generation, evaluation, and
  subsequent training. Check the relevant consumers. Do not imply CUDA and
  MLX checkpoint formats are interchangeable without verifying conversion.
- Keep necessary changes to MLX helper files local to the requested algorithm.
  Do not rewrite unrelated MLX algorithms, change canonical CUDA code to make
  the port easier, or modify generated checkpoints and TensorBoard runs.

## Verify the Conversion

1. Compare the completed files side by side for section order, names, formulas,
   branches, and loop/update structure. Account for each backend difference.
2. Compare parser contracts mechanically, including defaults and validation,
   for no arguments, representative CUDA commands, and invalid inputs. On
   hardware without MLX, inspect or isolate the parser with AST tools rather
   than importing the whole training entrypoint. Do not count static checks as
   runtime verification.
3. Run focused syntax and available lint checks using `uv`; run
   `git diff --check`. Avoid unrelated formatting changes.
4. Where hardware permits, compare small fixed-input calculations with CUDA
   or independently computed expected values using suitable dtype tolerances.
   Cover the changed loss, masks, normalization, gradient boundaries, and
   optimizer/schedule behavior as applicable. Expect numerical differences;
   acceptance of arguments alone does not establish algorithm parity.
5. On supported hardware, exercise a short debug/smoke configuration, including
   applicable validation and a temporary checkpoint save/load. Keep generated
   artifacts outside tracked source and existing experiment outputs. Do not
   launch long training unless explicitly requested.
6. Report the replaced/created files, passed checks, any necessary deviations,
   and checks unavailable on the current hardware. Include the equivalent MLX
   command. Do not claim runtime or artifact compatibility that was not tested.
