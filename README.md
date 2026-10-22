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

This trains a two-layer graph attention network over sentence nodes. Its answer
heads predict yes/no or an answer span, and its support head scores evidence
sentences. Training uses all configured examples and saves the best joint
answer/support checkpoint under `outputs/checkpoints/basic_gat/`. The default
device setting selects an available TPU, GPU, or CPU; use `--device cpu`,
`--device cuda`, or `--device xla` to choose explicitly. Training settings are
in `configs/models/basic_gat_frozen.yaml`.

The graph builder also supports a legacy clique graph. Train it with
`python src/train.py --graph-mode legacy`; outputs are saved in separate
`_legacy` directories. Compare graph structure and support-pair reachability
with `python src/analyze_graphs.py --split validation`.

### Transformer-only support baseline

To compare graph message passing with global transformer attention, run the
BERT-only support baseline:

```bash
python src/train_transformer_baseline.py
```

This model presents the question and candidate sentences as one BERT input, so
self-attention can connect evidence across sentences and paragraphs. It predicts
supporting facts directly and has no answer-span objective. It reports support
EM, precision, recall, and F1, matching the GAT's evidence metrics. BERT's
default input limit is 512 tokens; candidate sentences omitted by truncation are
tracked in validation predictions. Settings are in
`configs/models/transformer_support_baseline.yaml`; checkpoints and run outputs
are saved separately under `outputs/checkpoints/transformer_support/` and
`experiments/runs/transformer_support/`.

## Kaggle

The push utility packages the source and configs as a Kaggle script kernel,
enables internet access for Hugging Face downloads, and starts a run. TPU v5e-8
is the default accelerator; use `--gpu` to request a Kaggle NVIDIA T4 GPU, or
`--cpu` for CPU execution. Kaggle API credentials must be configured for the
Kaggle CLI.

```bash
python -m pip install kaggle
python scripts/push_to_kaggle.py --kernel-id YOUR_KAGGLE_USERNAME/MultihopQA
```

To push and run the BERT-only support baseline on the default TPU, add
`--transformer-baseline`:

```bash
python scripts/push_to_kaggle.py \
  --kernel-id YOUR_KAGGLE_USERNAME/MultihopQA \
  --transformer-baseline
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
