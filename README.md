# Graph-Enhanced Multi-Hop QA

This repository is the implementation workspace for the research plan in
`Documents/technical-plan-gat-hotpotqa.pdf`. The current code downloads and
prepares HotpotQA distractor data from Hugging Face. The prepared examples keep
the source sentence boundaries and supporting-fact labels. The first model is
the Basic GAT experiment.

## Prepare the complete dataset

From the repository's top-level folder, install the data preparation dependencies and run:

```bash
pip install -r requirements.txt
python src/prepare_hotpotqa.py
```

The first run downloads the dataset and stores the Hugging Face cache under
`data/raw/hotpotqa/huggingface_cache/`. It processes the complete `train` and
`validation` splits into:

- `data/processed/hotpotqa/train.jsonl`
- `data/processed/hotpotqa/validation.jsonl`
- `data/processed/hotpotqa/preparation_summary.json`

The dataset source and output folders are in
`configs/data/hotpotqa_distractor.yaml`. The defaults use all 90,447 training
examples and all 7,405 validation examples. Each JSONL line is one question,
with its answer, question type, difficulty, flattened sentence nodes, and a
supporting-fact label for every sentence. Each node keeps its paragraph index,
title, and original sentence index.

To check the processed files after preparation, run:

```bash
python src/audit_hotpotqa.py
```

## Train the first Basic GAT experiment

After preparing the data, run:

```bash
python src/train.py
```

This trains a two-layer sentence GAT with a frozen BERT encoder, answer-type
and span heads, and a supporting-fact head. BERT runs on each batch so the span
head receives token-level features; the graph layers and prediction heads are
trained. The initial graph connects sentences that share a paragraph title or
a simple detected proper name. The script uses all configured train and
validation examples, detects available GPUs, and saves the best checkpoint
under `outputs/checkpoints/basic_gat/`. The settings are in
`configs/models/basic_gat_frozen.yaml`.

For each question, the training code reads its sentence nodes and answer labels
from JSONL. The collator tokenizes the question together with each sentence,
builds the graph adjacency matrix, and pads graphs in a batch to a shared size.
BERT creates token features; the GAT layers pass information between linked
sentence nodes. The model then predicts `yes`, `no`, or a text span, and scores
each sentence as a supporting fact. Training compares these predictions with
the answer, span, and supporting-fact labels. During validation, the best span
is decoded from its token IDs and the support scores are mapped back to source
titles and sentence numbers. The best checkpoint also stores the full model
settings used for that run.

This first version keeps BERT frozen, but still runs it on every batch so the
span head can use token-level representations. That makes training more
compute-intensive than a sentence-vector-only model.

HotpotQA already supplies sentence segmentation. The preparation code uses those sentence strings as-is, so supporting-fact indices continue to refer to the correct sentences.

## Research scope

- HotpotQA distractor split, using the candidate paragraphs supplied by the dataset.
- One sentence-level graph per question, with self-loops and shared-entity edges only.
- A two-layer GAT model with a frozen BERT sentence encoder, supporting-fact prediction, extractive span prediction, and yes/no classification.
- A later comparison with joint BERT fine-tuning.
- Controlled self-loop-only, shuffled-edge, relation, depth, and head-count experiments.
- Answer, supporting-fact, and joint evaluation with hop-stratified and error analysis.

Retrieval and DFGN are outside the initial implementation scope.
