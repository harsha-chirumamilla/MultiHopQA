"""Train the sentence-graph QA model on CPU, CUDA, or a Kaggle TPU."""

import argparse
import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer

from multihopqa.data.collators import GraphQADataCollator
from multihopqa.data.dataset import HotpotJsonlDataset
from multihopqa.graph.builder import GRAPH_MODES
from multihopqa.models.graph_qa import GATForQuestionAnswering
from multihopqa.training.trainer import train_model
from multihopqa.utils.config import check_data_config
from multihopqa.utils.config import load_config
from multihopqa.utils.seed import set_seed


def resolve_device(requested_device):
    """Choose a backend without importing/initializing XLA in the parent process."""
    if requested_device == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError(
                "PyTorch/XLA is not installed. Select Kaggle's TPU runtime, which "
                "provides a torch_xla build compatible with its PyTorch version."
            )
        return "xla"

    if requested_device == "auto":
        tpu_environment = (
            os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
            or bool(os.environ.get("XRT_TPU_CONFIG"))
            or bool(os.environ.get("TPU_NAME"))
        )
        if tpu_environment:
            if importlib.util.find_spec("torch_xla") is None:
                raise RuntimeError(
                    "A TPU runtime was detected, but torch_xla is unavailable. "
                    "Use Kaggle's TPU image or run with --device cpu/cuda."
                )
            return "xla"
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return requested_device


def build_model_and_tokenizer(model_config, device, is_master=True):
    """Load the tokenizer and frozen encoder, then place the QA model."""
    model_name = model_config["model_name"]
    if is_master:
        print(f"Loading tokenizer and frozen text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name)
    model = GATForQuestionAnswering(
        encoder=encoder,
        graph_hidden_size=model_config["graph_hidden_size"],
        attention_heads=model_config["attention_heads"],
        dropout=model_config["dropout"],
    ).to(device)
    return model, tokenizer


def validate_prepared_data(data_config):
    """Return the prepared data paths after checking splits and files."""
    check_data_config(data_config)
    if (
        "train" not in data_config["splits"]
        or "validation" not in data_config["splits"]
    ):
        raise ValueError("Basic GAT training needs both train and validation splits.")

    data_dir = Path(data_config["output_dir"])
    train_file = data_dir / "train.jsonl"
    validation_file = data_dir / "validation.jsonl"
    if not train_file.exists() or not validation_file.exists():
        raise FileNotFoundError(
            "Prepared train and validation files are required. "
            "Run python src/prepare_hotpotqa.py first."
        )
    return train_file, validation_file


def build_data_loaders(
    train_file,
    validation_file,
    tokenizer,
    model_config,
    graph_mode,
    batch_size,
    train_sampler=None,
    include_validation=True,
    device_type="cpu",
):
    """Build loaders, optionally with static XLA shapes and a distributed sampler."""
    collator = GraphQADataCollator(
        tokenizer,
        model_config["max_sequence_length"],
        graph_mode=graph_mode,
        static_shapes=device_type == "xla",
    )
    train_dataset = HotpotJsonlDataset(train_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device_type == "cuda",
    )

    validation_loader = None
    validation_dataset = None
    if include_validation:
        validation_dataset = HotpotJsonlDataset(validation_file)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=device_type == "cuda",
        )
    return train_loader, validation_loader, len(train_dataset), validation_dataset


def apply_graph_mode(model_config, graph_mode):
    model_config["graph_mode"] = graph_mode
    if graph_mode == "legacy":
        model_config["checkpoint_dir"] = f"{model_config['checkpoint_dir']}_legacy"
        model_config["run_dir"] = f"{model_config['run_dir']}_legacy"
    return model_config


def run_single_device(
    arguments,
    model_config,
    train_file,
    validation_file,
    device_name,
):
    """Run the existing CPU/CUDA training path."""
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    gpu_count = torch.cuda.device_count() if device_name == "cuda" else 0
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs with DataParallel.")
    elif gpu_count == 1:
        print("Using one GPU.")
    else:
        print("Using CPU. Training will be slow.")

    set_seed(model_config["seed"])
    base_model, tokenizer = build_model_and_tokenizer(model_config, device)
    model = torch.nn.DataParallel(base_model) if gpu_count > 1 else base_model
    (
        train_loader,
        validation_loader,
        train_count,
        validation_dataset,
    ) = build_data_loaders(
        train_file,
        validation_file,
        tokenizer,
        model_config,
        arguments.graph_mode,
        model_config["batch_size"],
        device_type=device_name,
    )
    print(
        f"Using complete splits: {train_count:,} train and "
        f"{len(validation_dataset):,} validation examples."
    )
    train_model(model, train_loader, validation_loader, device, tokenizer, model_config)


def xla_worker(index, arguments, model_config, train_file, validation_file):
    """Run one PyTorch/XLA worker on its assigned TPU process/device."""
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as parallel_loader
    import torch_xla.runtime as xr

    device_type = xr.device_type()
    if device_type != "TPU":
        raise RuntimeError(
            f"PyTorch/XLA reported device type {device_type!r}, not TPU. "
            "The Kaggle push runtime may not have assigned a TPU accelerator."
        )

    device = xm.xla_device()
    rank = index
    world_size = xr.world_size()
    is_master = rank == 0
    global_batch_size = model_config["batch_size"]
    if global_batch_size % world_size:
        raise ValueError(
            f"Configured global batch_size ({global_batch_size}) must be divisible "
            f"by the XLA worker count ({world_size})."
        )
    local_batch_size = global_batch_size // world_size

    set_seed(model_config["seed"])
    if is_master:
        print(
            f"Using Kaggle TPU with {world_size} XLA workers; "
            f"global batch size {global_batch_size} ({local_batch_size} per worker)."
        )

    base_model, tokenizer = build_model_and_tokenizer(
        model_config,
        device,
        is_master=is_master,
    )
    train_dataset = HotpotJsonlDataset(train_file)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=model_config["seed"],
        drop_last=False,
    )
    collator = GraphQADataCollator(
        tokenizer,
        model_config["max_sequence_length"],
        graph_mode=arguments.graph_mode,
        static_shapes=True,
    )
    cpu_train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )
    train_loader = parallel_loader.MpDeviceLoader(cpu_train_loader, device)

    validation_loader = None
    if is_master:
        validation_dataset = HotpotJsonlDataset(validation_file)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
        print(
            f"Using complete splits: {len(train_dataset):,} train and "
            f"{len(validation_dataset):,} validation examples."
        )

    train_model(
        base_model,
        train_loader,
        validation_loader,
        device,
        tokenizer,
        model_config,
        xla_model=xm,
        rank=rank,
        train_sampler=train_sampler,
    )


def parse_args():
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
    parser.add_argument(
        "--graph-mode",
        choices=GRAPH_MODES,
        default="current",
        help="Graph construction to train with; use legacy for the original graph.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xla"),
        default="auto",
        help="Training backend. Auto prefers a detected TPU, then CUDA, then CPU.",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    data_config = load_config(arguments.data_config)
    model_config = apply_graph_mode(
        load_config(arguments.model_config),
        arguments.graph_mode,
    )
    print(f"Training with graph mode: {arguments.graph_mode}")
    train_file, validation_file = validate_prepared_data(data_config)
    device_name = resolve_device(arguments.device)

    if device_name == "xla":
        import torch_xla.distributed.xla_multiprocessing as xmp

        # Kaggle's TPU image exports "local" as a single host address. PJRT
        # interprets it as a worker list and cannot initialize the v5e-8 slice
        # when xmp.spawn starts one process per TPU core.
        if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
            os.environ.pop("TPU_PROCESS_ADDRESSES")

        print("Launching PyTorch/XLA workers.")
        xmp.spawn(
            xla_worker,
            args=(arguments, model_config, train_file, validation_file),
            start_method="spawn",
        )
        return

    run_single_device(
        arguments,
        model_config,
        train_file,
        validation_file,
        device_name,
    )


if __name__ == "__main__":
    main()
