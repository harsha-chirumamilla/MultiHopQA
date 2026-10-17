"""Train the first sentence-graph QA model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer

from multihopqa.data.collators import GraphQADataCollator
from multihopqa.data.dataset import HotpotJsonlDataset
from multihopqa.models.graph_qa import BasicGATForQuestionAnswering
from multihopqa.training.trainer import train_model
from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config
from multihopqa.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train the Basic GAT QA model.")
    parser.add_argument(
        "--data-config",
        default="configs/data/hotpotqa_distractor.yaml",
        help="Dataset config path, relative to the repository folder.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/basic_gat_frozen.yaml",
        help="Basic GAT training config path, relative to the repository folder.",
    )
    arguments = parser.parse_args()

    data_config = load_config(arguments.data_config)
    model_config = load_config(arguments.model_config)
    check_data_config(data_config)

    if "train" not in data_config["splits"] or "validation" not in data_config["splits"]:
        raise ValueError("Basic GAT training needs both train and validation splits.")

    set_seed(model_config["seed"])
    data_dir = Path(data_config["output_dir"])
    train_file = data_dir / "train.jsonl"
    validation_file = data_dir / "validation.jsonl"
    if not train_file.exists() or not validation_file.exists():
        raise FileNotFoundError(
            "Prepared train and validation files are required. "
            "Run python src/prepare_hotpotqa.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs with DataParallel.")
    elif gpu_count == 1:
        print("Using one GPU.")
    else:
        print("Using CPU. Training will be slow.")

    model_name = model_config["model_name"]
    print(f"Loading tokenizer and frozen text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name)

    base_model = BasicGATForQuestionAnswering(
        encoder=encoder,
        graph_hidden_size=model_config["graph_hidden_size"],
        attention_heads=model_config["attention_heads"],
        dropout=model_config["dropout"],
    )
    base_model.to(device)

    if gpu_count > 1:
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    collator = GraphQADataCollator(
        tokenizer,
        model_config["max_sequence_length"],
    )
    train_dataset = HotpotJsonlDataset(train_file)
    validation_dataset = HotpotJsonlDataset(validation_file)
    print(
        f"Using complete splits: {len(train_dataset):,} train and "
        f"{len(validation_dataset):,} validation examples."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    train_model(
        model,
        train_loader,
        validation_loader,
        device,
        tokenizer,
        model_config,
    )


if __name__ == "__main__":
    main()
