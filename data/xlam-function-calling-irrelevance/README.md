---
pretty_name: xLAM Tool Calling with Abstention
language:
- en
tags:
- tool-calling
- function-calling
- abstention
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
---

# xLAM Tool Calling with Abstention

Training only on examples that require a tool call can encourage a model to call a tool even when none of the available tools is suitable. This dataset combines valid tool-call examples with irrelevant-tool no-call examples to teach both call construction and when to abstain.

The positive examples come from [`lockon/xlam-function-calling-60k`](https://huggingface.co/datasets/lockon/xlam-function-calling-60k) and are normalized and split before mixing. Tool schemas are converted to Transformers' standard function schema, including nested parameter types, descriptions, and defaults. Malformed rows, unavailable reference tools, unknown reference arguments, and duplicate tool names are discarded. Queries are grouped after collapsing whitespace before splitting. The negative examples come from [`MadeAgents/xlam-irrelevance-7.5k`](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k). MadeAgents constructed it by removing the ground-truth tool from sampled xLAM examples and setting `answers` to `[]`.

Both sources are normalized to the same JSONL schema: `id`, `query`, JSON-encoded `tools`, and JSON-encoded `answers`. Positive IDs start with `xlam:` and negative IDs with `irrelevance:`. Every retained row has at least one available tool. Negative examples retain irrelevant offered tools and target no tool call (`[]`); irrelevance rows with no tools are excluded. Rows with malformed tools, duplicate tool names, or nonempty irrelevance answers are discarded.

The target test fraction is 5% of each source. xLAM uses its original seeded, whitespace-normalized query-group split. Every negative row whose query occurs in xLAM follows that query into train or test; unmatched negative queries are split by group toward the 5% target. This prevents a positive and negative version of a query from crossing the train/test boundary. The negative test fraction may differ slightly from exactly 5%; the manifest records actual counts. Train and test are each shuffled after mixing.

With the pinned source revisions and default seed, preparation retains 59,740 positive rows and 6,034 irrelevant-tool no-call rows. The mixed files contain 56,753 calls plus 5,732 no-calls for training, and 2,987 calls plus 302 no-calls for testing. Of the 7,500 irrelevance source rows, 1,446 have no available tools and are excluded; 20 have duplicate available-tool names and are excluded. The manifest records these exclusions and the generated file hashes.

## Training and evaluation

For training and evaluation code and instructions, see [Post-Training Lab](https://github.com/uygarkurt/post-training-lab).

## Dataset files

The prepared splits are stored in `train.jsonl` and `test.jsonl`. The accompanying `manifest.json` records source revisions, source and output hashes, row counts, filtering exclusions, the random seed, and split settings.
