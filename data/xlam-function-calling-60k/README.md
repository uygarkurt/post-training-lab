# xLAM function calling: local train/test split

Rebuild the prepared data from the repository root (CPU only):

```bash
uv run python data/xlam-function-calling-60k/prepare.py
```

The script downloads the single training file from
[`lockon/xlam-function-calling-60k`](https://huggingface.co/datasets/lockon/xlam-function-calling-60k),
normalizes its tool definitions, and saves `train.jsonl`, `test.jsonl`, and
`manifest.json` beside the script. The source revision and file hash are pinned.
Rerunning replaces these generated files. Use `--output-dir /tmp/xlam-check`
to build a separate copy, or `--source-file` with an existing source download.

## Preparation and split

The source has 60,000 single-turn synthetic function-calling examples. Its
`tools` and `answers` fields contain JSON serialized as strings. Preparation:

1. Parses both fields and rejects rows with an empty query, no available tool,
   no reference call, malformed call arguments, or a reference to an unavailable
   tool. It also rejects tool lists that reuse the same function name.
2. Converts each source tool to the standard Transformers JSON schema:
   `{"type":"function","function":{"name":...,"parameters":...}}`.
   Python-like source parameter types, including nested collection element
   types, are mapped to JSON Schema. A parameter is inferred to be optional
   when its source type contains
   `optional` or it has a default; all others enter the schema's `required`
   list. Descriptions and explicit defaults are preserved.
3. Keeps the normalized tools and answer calls JSON-encoded inside each JSONL
   row. The runtime loader decodes them. This avoids treating thousands of
   arbitrary parameter names as tabular columns.
4. Groups queries after collapsing whitespace, shuffles groups with seed 42,
   assigns 5% of rows (rounded up) to test without separating any group, and
   shuffles each output split.

The default recipe removes 260 rows whose available tools reuse a function
name, including 89 rows where the repeated name has different parameter
signatures. Standard calls identify tools by name, so those rows would be
ambiguous after conversion. The resulting split contains **56,753 train /
2,987 test** examples and 57,925 whitespace-normalized query groups. Exact
queries cannot cross the train/test boundary. No token-length filter is applied
during preparation; prompts are filtered against the selected model at runtime.
The manifest records revisions, hashes, normalization rules, split settings,
counts, and software versions.

This is a local holdout from the source training corpus, not an official xLAM
test benchmark. Every retained row requires at least one call, so it does not
measure whether a model can correctly decline to use tools. The source is
synthetic; exact-match scores should be paired with inspection of saved model
outputs before drawing conclusions about general tool-use quality.

## Evaluation

Evaluate the base model on a repeatable subset:

```bash
uv run --extra cuda -m cuda_backend.xlam_function_calling_eval \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --num-samples 100 \
  --output-jsonl data/outputs/xlam_base.jsonl
```

For an SFT adapter, add `--load-adapter` and point `--model_path` to the adapter
checkpoint. Omit `--num-samples` to evaluate all usable test rows. The evaluator
loads only `test.jsonl`, renders the query and tools through the tokenizer's
native tool-aware chat template, left-pads prompts, and generates greedily.
A tokenizer whose template ignores tools is rejected rather than evaluated
with an incomplete prompt.

The primary metric is exact tool-call-set accuracy: the predicted and reference
calls must have identical function names, argument keys, JSON value types, and
values. Object key order and parallel-call order are ignored; array order and
duplicate calls are preserved. Function-name-set accuracy is also reported to
separate tool selection from argument construction, together with parse and
token-limit counts. Qwen-style `<tool_call>` blocks and bare JSON object/list
responses are accepted. Extra calls, missing calls, malformed JSON, or prose
without a parseable call count as incorrect.

The JSONL output stores the source ID, query, available tools, references, raw
completion, parsed calls, both correctness flags, and truncation. Each run
overwrites the output file.

With the Qwen2.5-0.5B-Instruct tokenizer at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, all 2,987 test prompts fit the
default 3,072-token prompt limit: median 459, 95th percentile 954, and maximum
2,176 tokens. Complete prompt-plus-reference examples have a maximum of 2,224
tokens in this test split, and the assistant reference alone has a maximum of
620 tokens, below the evaluator's 1,024-token completion limit.

## Supervised fine-tuning

xLAM remains available for CUDA SFT with `--dataset xlam-function-calling`.
With the Qwen tokenizer revision measured above, a 2,048-token limit
retains 56,734 of 56,753 prepared training
rows. The 19 longer training rows are discarded rather than truncated. Recheck
this distribution if the model or chat template changes:

```bash
uv run --extra cuda -m cuda_backend.sft_train \
  --dataset xlam-function-calling \
  --max-seq-len 2048
```

The loader reads only `train.jsonl`, uses the tokenizer's tool-aware chat
template, and computes loss only on the assistant tool calls. Its seeded
runtime validation subset comes from the prepared training file. The prepared
test file is never loaded during SFT. GRPO integration is not included.
