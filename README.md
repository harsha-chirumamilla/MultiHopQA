# Graph-Enhanced Multi-Hop QA

This project trains and evaluates a graph neural network for multi-hop question
answering on HotpotQA. It represents each question's candidate evidence as a
graph of sentences, encodes the question and sentences with BERT, and predicts
an answer, supporting facts, and (when applicable) an answer span.

## Dataset

The data pipeline downloads the HotpotQA distractor dataset from Hugging Face
and saves the configured `train` and `validation` splits as JSONL. It preserves
the dataset's sentence boundaries and supporting-fact annotations.

Install the dependencies and prepare the data from the repository root:

```bash
pip install -r requirements.txt
python src/prepare_hotpotqa.py
```

The first run caches the source dataset in
`data/raw/hotpotqa/huggingface_cache/` and writes:

- `data/processed/hotpotqa/train.jsonl`
- `data/processed/hotpotqa/validation.jsonl`
- `data/processed/hotpotqa/preparation_summary.json`

By default, this includes 90,447 training examples and 7,405 validation
examples. Each JSONL record contains a question, answer, question type,
difficulty, sentence nodes, and supporting-fact labels. Sentence nodes retain
their paragraph title and original sentence index. Dataset settings are in
`configs/data/hotpotqa_distractor.yaml`.

To inspect the prepared data, run:

```bash
python src/audit_hotpotqa.py
```

## Model and training

The model uses a frozen BERT encoder to produce token features and a two-layer
graph attention network to pass information between sentence nodes. The graph
connects sentences to an article hub and includes links based on normalized
title mentions and repeated name phrases. Prediction heads classify `yes`,
`no`, or span answers, and score sentences as supporting facts.

Start training after preparing the dataset:

```bash
python src/train.py
```

## Kaggle

The push utility packages the source and configs as a Kaggle script kernel,
enables internet access for Hugging Face downloads, and starts a run. TPU v5e-8
is the default accelerator; use `--cpu` to request CPU execution. Kaggle API
credentials must be configured for the Kaggle CLI.

```bash
python -m pip install kaggle
python scripts/push_to_kaggle.py --kernel-id YOUR_KAGGLE_USERNAME/MultihopQA
```

The kernel prepares HotpotQA if the processed splits are missing. Training
artifacts are written under
`/kaggle/working/multihopqa_project/outputs` and can be downloaded with:

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/MultihopQA
```

Use `--public` to publish the kernel or `--staging-dir PATH` to keep the
generated upload files locally. TPU runs require PyTorch/XLA in the Kaggle TPU
image. For an interactive Kaggle notebook, select a TPU accelerator and run
`python src/train.py --device auto` (or `--device xla`). The push utility also
accepts `KAGGLE_KERNEL_ID=owner/slug` instead of `--kernel-id`.

## Evaluation and diagnostics

The project includes scripts for answer and supporting-fact evaluation,
HotpotQA-specific audits, and graph analysis.