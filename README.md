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

The joint reader takes the question and all titled evidence sentences. It splits
long contexts into overlapping BERT windows so every supplied sentence can
contribute. It predicts yes/no or an answer span, scores supporting facts, and
restricts span decoding to its two highest-scoring evidence sentences.

The reference model fine-tunes BERT over the context windows:

```bash
python src/train.py --model-config configs/models/bert_reader.yaml
```

The graph-enhanced model uses the same windows, encoder, supervision, loss, and
evaluation, adding two graph-attention layers over sentence representations:

```bash
python src/train.py --model-config configs/models/gat_reader.yaml
```

Both configurations report answer EM/F1, supporting-fact EM/F1, and joint
metrics. They save the best joint-F1 checkpoint, training history, and
validation predictions to their own directories. Run on an available TPU, GPU,
or CPU with `--device auto` (default), or select one explicitly with
`--device xla`, `--device cuda`, or `--device cpu`. The configs use a conservative
batch size because one question may require several 512-token windows.

The graph builder supports `current` and `legacy` edge construction. Use
`--graph-mode legacy` for the latter; its outputs are saved in separate
`_legacy` directories. Compare graph structure and support-pair reachability
with `python src/analyze_graphs.py --split validation`.

### Support-only baseline

`python src/train_transformer_baseline.py` runs the earlier BERT model that
predicts supporting facts only. It is useful for evidence-selection experiments,
but it does not provide an answer-score comparison with the joint readers.

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

To push and run the joint GAT reader on the default TPU, select `--reader gat`:

```bash
python scripts/push_to_kaggle.py \
  --kernel-id YOUR_KAGGLE_USERNAME/MultihopQA \
  --reader gat
```

The default `--reader bert` runs the fine-tuned BERT reference model. Add
`--transformer-baseline` to run the older support-only experiment.

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
