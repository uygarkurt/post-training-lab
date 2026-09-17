# xLAM function calling with Hammer irrelevance examples

Prepare a mixed SFT training file and a mixed local test file from the two
source datasets:

```bash
uv run python data/xlam-function-calling-irrelevance/prepare.py
```

The script writes `train.jsonl`, `test.jsonl`, and `manifest.json` in this
directory. Generated files are ignored by Git. Use `--output-dir` to write
elsewhere, or `--xlam-source-file` and `--irrelevance-source-file` to use local
copies of the source JSON arrays. The source revisions are pinned in the script;
the manifest records source and output hashes, counts, and split settings.
Rerunning the command replaces these three generated files.

The positive examples come from
[`lockon/xlam-function-calling-60k`](https://huggingface.co/datasets/lockon/xlam-function-calling-60k)
and use the same normalization and query-group split as the
[original xLAM preparation](../xlam-function-calling-60k/README.md).
The negative examples come from
[`MadeAgents/xlam-irrelevance-7.5k`](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k),
the irrelevance dataset released with Lin, Qiqiang, et al.,
“[Hammer: Robust Function-Calling for On-Device Language Models via Function
Masking](https://arxiv.org/abs/2410.04587),” *arXiv:2410.04587* (2024).
MadeAgents constructed it by removing the ground-truth tool from sampled xLAM
examples and setting `answers` to `[]`.

Both sources are normalized to the same JSONL schema: `id`, `query`,
JSON-encoded `tools`, and JSON-encoded `answers`. Positive IDs start with
`xlam:` and negative IDs with `irrelevance:`. Every retained row has at least
one available tool. Negative examples retain irrelevant offered tools and
target no tool call (`[]`); irrelevance rows with no tools are excluded so
no-tools prompts are not supervised to produce `[]`. Rows with malformed
tools, duplicate tool names, or nonempty irrelevance answers are discarded.

The target test fraction is 5% of each source. xLAM uses its original seeded,
whitespace-normalized query-group split. Every negative row whose query occurs
in xLAM follows that query into train or test; unmatched negative queries are
split by group toward the 5% target. This prevents a positive and negative
version of a query from crossing the train/test boundary. The negative test
fraction may differ slightly from exactly 5%; the manifest records actual
counts. Train and test are each shuffled after mixing.

With the pinned source revisions and default seed, preparation retains 59,740
positive rows and 6,034 irrelevant-tool no-call rows. The mixed files contain
56,753 calls plus 5,732 no-calls for training, and 2,987 calls plus 302
no-calls for testing. Of the 7,500 irrelevance source rows, 1,446 have no
available tools and are excluded; 20 have duplicate available-tool names and
are excluded. The manifest records these exclusions and the generated file
hashes. Length filtering occurs later in the SFT loader, so inspect its
retained and overlong counts for the chosen tokenizer and sequence limit.

## CUDA SFT and evaluation

This mixed dataset is the default for CUDA SFT. Specify
`--dataset xlam-function-calling` to train on the original positive-only split.

```bash
uv run --extra cuda -m cuda_backend.sft_train \
  --dataset xlam-function-calling-irrelevance \
  --max-seq-len 896

uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path checkpoints/cuda/sft/YOUR_RUN/step_XXXXXX \
  --load-adapter \
  --dataset-path data/xlam-function-calling-irrelevance/test.jsonl
```

SFT uses the model's tool-aware chat template. Positive targets remain native
assistant tool calls; negative targets are ordinary assistant messages
containing `[]`, with no tool call. The prompt is masked from the loss in both
cases. No-tools chat behavior is outside this tool-routing dataset; it must be
checked separately after training.
The existing SFT trainer takes a further seeded 5% runtime validation split
from the prepared training file; this row-level split may separate two
versions of the same query. Use the prepared mixed test file for held-out
evaluation.

The evaluator reports overall exact call-set accuracy, call-required accuracy,
and irrelevant-tool no-call accuracy. The regenerated local test file has
3,289 prompts; the evaluator applies its prompt cap when loading them. This is
a holdout from synthetic training sources, not an official BFCL test.
This preparation is for SFT; the existing GRPO dataset remains the original
positive-only xLAM training file.

The earlier mixed SFT pilot used a previous preparation that included no-tool
rows targeted to `[]`. Its checkpoint was not trained on the corrected files
and should be reported with that earlier dataset version.
