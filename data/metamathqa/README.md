# MetaMathQA: preprocessing and local train/test splits

Rebuild from the repository root (CPU only):

```bash
uv run --extra eval python data/metamathqa/prepare.py
```

**Queries are deduplicated by default, and only rows whose reference answers
can be parsed by Math-Verify are saved.** No 512-token filter is applied here;
sequence-length filtering remains a separate SFT step.

## Source and preprocessing order

The source is [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA),
file `MetaMathQA-395K.json`, from its single training split. The recipe pins
revision `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`. The source file's SHA-256 is
`fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725`.
All 395,000 source rows and all source `type` categories enter preprocessing.

The script performs these operations in this order:

1. **Extract the reference answer.** Set `ground_truth` to everything after
   the last exact, case-sensitive `The answer is:` marker in `response`,
   removing surrounding whitespace. Preserve `response` unchanged. A missing
   marker stops preparation with an error; the pinned source has no missing
   markers. The script does not repair answer text or infer missing answers.
2. **Drop `original_question`.** It is not used for grouping, splitting, or
   prompting. Saved rows contain `type`, `query`, `response`, and `ground_truth`.
3. **Deduplicate exact queries by default.** Use pandas `drop_duplicates` on
   `query`, keeping the first row in source order, including its response.
   Matching is case-sensitive and does not normalize whitespace. Rows with the
   same query but different responses count as duplicates. Deduplication occurs
   before reference filtering: a later duplicate is not substituted if the
   first row's reference fails to parse. `--keep-duplicate-queries` skips only
   this step.
4. **Filter reference answers with Math-Verify.** Reject empty `ground_truth`
   values. For other values, wrap the raw answer in dollar-sign math delimiters
   and call `math_verify.parse` with `LatexExtractionConfig()`,
   `fallback_mode="no_fallback"`, `extraction_mode="any_match"`,
   `parsing_timeout=5`, and `raise_on_error=False`. Retain the row only if parsing
   returns a nonempty result. These settings match the evaluator's reference
   parser. Failed or timed-out parses are discarded. Parsed expressions are
   used only for this decision; the saved `ground_truth` remains raw text.
5. **Create fresh shuffled train/test splits.** Use Hugging Face
   `Dataset.train_test_split(test_size=0.05, seed=42)` on the retained rows.
   Test receives `ceil(0.05 × retained rows)` and train receives the rest.
   The library uses NumPy's seeded default generator (PCG64) to permute rows.
   Save UTF-8 JSONL files and a provenance manifest beside `prepare.py`.

No subject selection, tokenization, sequence-length truncation, semantic
deduplication, or response-correctness verification is performed during this
preparation. Empty queries are not separately filtered at this stage; the
evaluation loader rejects them. A successful parse means the reference can be
read as a mathematical expression, not that it correctly solves the problem.

## Measured counts for the default recipe

| Stage | Removed at this stage | Rows remaining |
|---|---:|---:|
| Source | — | 395,000 |
| Exact-query deduplication | 248,884 | 146,116 |
| Math-Verify reference filter | 185 | 145,931 |
| Final train split (95%) | — | **138,634** |
| Final test split (5%, rounded up) | — | **7,297** |

The 185 reference-filter removals comprise **7 empty answers and 178 nonempty
answers that failed parsing**. Every saved reference passes the evaluator's
parser under the recorded software versions. These counts precede any
model-specific SFT length filter or runtime train/validation holdout.

This is a new split of the filtered source, replacing the earlier split made
before reference filtering. Its membership changes even though the seed stays
42; results on the earlier test file are not results on this new test file.

## Options and reproducibility

To keep repeated queries while still filtering unparseable references:

```bash
uv run --extra eval python data/metamathqa/prepare.py --keep-duplicate-queries
```

With this option and the same pinned source/software, the parser removes 702
rows (including 19 empty answers), retaining 394,298 rows: **374,583 train /
19,715 test**. These are alternative outputs; the files in this directory use
the default deduplication recipe above.

Use `--test-fraction` and `--seed` to change the split settings, and
`--output-dir /tmp/metamathqa-check` to generate a separate copy. Rerunning into
the same directory overwrites `train.jsonl`, `test.jsonl`, and `manifest.json`.
`--source-file /path/to/MetaMathQA-395K.json` uses an existing source download.
Reference filtering is always applied; it has no disable flag.

The [manifest](manifest.json) records the source revision and hash, script hash,
parser settings, counts at each stage, deduplication flag, split seed/fraction,
Python/package versions, and output SHA-256 hashes. A local source override is
identified by its path and hash instead of claiming a Hub revision. Reproduce
the recorded counts and output bytes with the pinned source and the recorded
software versions:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| datasets | 4.8.4 |
| pandas | 3.0.2 |
| NumPy | 2.4.4 |
| PyArrow | 23.0.1 |
| huggingface-hub | 1.10.1 |
| math-verify | 0.9.0 |
| latex2sympy2-extended | 1.11.0 |
| SymPy | 1.14.0 |
| antlr4-python3-runtime | 4.13.2 |

With default deduplication, exact queries cannot cross the train/test boundary.
With duplicates retained, they can. Different queries derived from the same
original problem can cross the boundary in either mode because
`original_question` is deliberately ignored. The local holdout is therefore
an exact-query-disjoint split by default, not a split of original-problem
families or an official MetaMathQA test benchmark.

## Methods text for the paper

We prepared a local train/test split from the 395,000-example MetaMathQA source
at revision `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`. We extracted reference
answers from the text following the final `The answer is:` marker, trimmed
surrounding whitespace, and discarded `original_question`. We deduplicated
exact query strings, retaining the first source occurrence, which left 146,116
examples. We then excluded examples with empty reference answers or references
that Math-Verify 0.9.0 could not parse as LaTeX with string fallback disabled.
This removed 185 examples and left 145,931. We randomly partitioned the retained
examples using seed 42, assigning 5% rounded up to test, yielding 138,634
training and 7,297 test examples. No token-length or subject filter was applied
during dataset preparation. Reference parseability was used as a formatting
criterion and does not establish the correctness of the supplied solutions.

## Evaluation

```bash
uv run --extra cuda --extra eval -m cuda_backend.metamathqa_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-path data/metamathqa/test.jsonl \
  --num-samples 100 \
  --output-jsonl metamathqa_eval.jsonl
```

Evaluate a PEFT adapter checkpoint with the same test settings:

```bash
uv run --extra cuda --extra eval -m cuda_backend.metamathqa_eval \
  --model_path ./checkpoints/cuda/sft/sft_<timestamp>/step_000500 \
  --load-adapter \
  --num-samples 100 \
  --output-jsonl metamathqa_adapter.jsonl
```

Omit `--num-samples` to evaluate all usable test rows. Omit `--load-adapter`
for a base model or full-model checkpoint. The default dataset path
is this directory's `test.jsonl`, resolved independently of the working directory.
The evaluator never loads the training file.

Evaluation uses a user message containing only `query`, greedy batched
generation, and Math-Verify from the existing `eval` extra. Defaults match
NuminaMath evaluation: Qwen2.5-0.5B-Instruct, batch size 8, maximum prompt
length 512, and maximum completion length 512. Prompts are left-padded.
Empty queries, overlong prompts, and unparseable reference answers are skipped
and reported separately as well as in the total skipped count. The sample limit
selects the first N usable rows after scanning the test file; prompts are never
truncated.

For the regenerated default split and the default Qwen2.5-0.5B-Instruct
tokenizer at a 512-token prompt limit, the loader retains **7,293 of 7,297**
test rows: 4 prompts are overlong, 0 references fail parsing, and 0 queries
are empty. This is a data-loading check; it does not report model accuracy.

For completions containing `The answer is:`, grading uses the final suffix.
Otherwise it uses Math-Verify's LaTeX/numeric extraction, prioritizing boxed
answers. Bare LaTeX references are wrapped in math delimiters for parsing.
Grading accepts mathematically equivalent values, such as `0.5` and `1/2`.
String fallback is disabled; unparseable predictions count as incorrect.
Accuracy is measured over usable rows, and token-limit truncations are reported.
The optional JSONL output stores each question, raw and parsed reference answer,
completion, parsed model answer, correctness, and truncation flag. Each invocation
overwrites it. `answer` is the raw reference text; `ground_truth` is its parsed
representation. `predicted_answer` contains the exact parsed model expressions
used in the comparison, serialized as a list of strings, just like
`ground_truth`. For example, a model completion ending in `\boxed{92}` produces
`"predicted_answer": ["92"]`. A failed extraction produces `[]` and counts as
incorrect. Grading compares the parsed reference with the parsed prediction
once; it does not also compare the raw `answer` separately.

Only evaluation dataset support is implemented here; SFT/GRPO integration is
left for a later change. Keep `test.jsonl` out of subsequent training.
