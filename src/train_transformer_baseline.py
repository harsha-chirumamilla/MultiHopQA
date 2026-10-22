"""Train the BERT-only global-context support-fact baseline."""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from multihopqa.data.dataset import HotpotJsonlDataset
from multihopqa.data.transformer_collator import TransformerSupportCollator
from multihopqa.models.transformer_support import TransformerSupportModel
from multihopqa.training.transformer_support import train_transformer_support_model
from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config
from multihopqa.utils.seed import set_seed
from train import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT-only HotpotQA support-fact baseline."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/hotpotqa_distractor.yaml",
        help="Dataset config path, relative to the repository folder.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/transformer_support_baseline.yaml",
        help="Baseline model config path, relative to the repository folder.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xla"),
        default="auto",
        help="Training backend. Auto selects an available TPU, GPU, or CPU.",
    )
    return parser.parse_args()


def xla_worker(index, train_path, validation_path, model_config):
    """Train one replica on a TPU core and validate on the master replica."""
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as parallel_loader
    import torch_xla.runtime as xr
    from torch.utils.data.distributed import DistributedSampler

    device = xm.xla_device()
    world_size = xr.world_size()
    if model_config["batch_size"] % world_size:
        raise ValueError(
            f"Configured global batch_size ({model_config['batch_size']}) must be "
            f"divisible by the XLA worker count ({world_size})."
        )
    local_batch_size = model_config["batch_size"] // world_size
    is_master = index == 0

    set_seed(model_config["seed"])
    if is_master:
        print(
            f"Using TPU with {world_size} XLA workers for the transformer baseline; "
            f"global batch size {model_config['batch_size']} "
            f"({local_batch_size} per worker)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], use_fast=True)
    encoder = AutoModel.from_pretrained(model_config["model_name"])
    model = TransformerSupportModel(
        encoder,
        freeze_encoder=model_config["freeze_encoder"],
        dropout=model_config["dropout"],
    ).to(device)
    collator = TransformerSupportCollator(
        tokenizer,
        max_sequence_length=model_config["max_sequence_length"],
    )

    train_dataset = HotpotJsonlDataset(train_path)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=index,
        shuffle=True,
        seed=model_config["seed"],
        drop_last=False,
    )
    cpu_train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    train_loader = parallel_loader.MpDeviceLoader(cpu_train_loader, device)

    validation_loader = None
    if is_master:
        validation_dataset = HotpotJsonlDataset(validation_path)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=model_config["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
        print(
            f"Using complete splits: {len(train_dataset):,} train and "
            f"{len(validation_dataset):,} validation examples."
        )

    train_transformer_support_model(
        model,
        train_loader,
        validation_loader,
        device,
        model_config,
        xla_model=xm,
        rank=index,
        train_sampler=train_sampler,
    )


def main():
    arguments = parse_args()
    data_config = load_config(arguments.data_config)
    check_data_config(data_config)
    model_config = load_config(arguments.model_config)
    if "train" not in data_config["splits"] or "validation" not in data_config["splits"]:
        raise ValueError("Training needs both train and validation splits.")

    data_dir = Path(data_config["output_dir"])
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Prepared train and validation JSONL files are required. "
            "Run python src/prepare_hotpotqa.py first."
        )

    device_name = resolve_device(arguments.device)
    if device_name == "xla":
        # Kaggle's TPU image exports a single-host placeholder that interferes
        # with PJRT's eight-worker initialization.
        if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
            os.environ.pop("TPU_PROCESS_ADDRESSES")
        import torch_xla.distributed.xla_multiprocessing as xmp

        print("Launching transformer baseline on TPU workers.")
        xmp.spawn(
            xla_worker,
            args=(train_path, validation_path, model_config),
            start_method="spawn",
        )
        return

    device = torch.device("cuda" if device_name == "cuda" else "cpu")

    set_seed(model_config["seed"])
    print(f"Loading BERT tokenizer and encoder: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], use_fast=True
    )
    encoder = AutoModel.from_pretrained(model_config["model_name"])
    model = TransformerSupportModel(
        encoder,
        freeze_encoder=model_config["freeze_encoder"],
        dropout=model_config["dropout"],
    ).to(device)

    collator = TransformerSupportCollator(
        tokenizer,
        max_sequence_length=model_config["max_sequence_length"],
    )
    train_dataset = HotpotJsonlDataset(train_path)
    validation_dataset = HotpotJsonlDataset(validation_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print(
        f"Training BERT support baseline on {device_name}: "
        f"{len(train_dataset):,} train and {len(validation_dataset):,} validation "
        "examples. This run predicts supporting facts; it does not train an answer "
        "span head."
    )
    train_transformer_support_model(
        model,
        train_loader,
        validation_loader,
        device,
        model_config,
        xla_model=None,
    )


if __name__ == "__main__":
    main()
