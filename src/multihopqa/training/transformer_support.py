"""Training and support-fact evaluation for the transformer-only baseline."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from multihopqa.evaluation.support_metrics import support_scores


MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "sentence_token_mask",
    "node_mask",
)


def _device_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _support_loss(logits, labels, node_mask):
    element_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    positives = node_mask & (labels > 0.5)
    negatives = node_mask & ~positives
    positive_loss = (element_losses * positives).sum() / positives.sum().clamp(min=1)
    negative_loss = (element_losses * negatives).sum() / negatives.sum().clamp(min=1)
    return (positive_loss + negative_loss) / 2.0


def _predicted_facts(logits, node_mask, metadata, threshold):
    selected = (torch.sigmoid(logits) >= threshold) & node_mask
    return [
        metadata[index]
        for index, is_selected in enumerate(selected.tolist())
        if is_selected
    ]


def evaluate_support_model(model, data_loader, device, config):
    model.eval()
    totals = {
        "support_em": 0.0,
        "support_precision": 0.0,
        "support_recall": 0.0,
        "support_f1": 0.0,
    }
    predictions = []
    examples_seen = 0
    threshold = config["support_threshold"]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validate support facts"):
            batch = _device_batch(batch, device)
            outputs = model(**{key: batch[key] for key in MODEL_INPUTS})
            logits = outputs["support_logits"].cpu()
            node_mask = batch["node_mask"].cpu()

            for index, example_id in enumerate(batch["example_ids"]):
                predicted = _predicted_facts(
                    logits[index],
                    node_mask[index],
                    batch["node_metadata"][index],
                    threshold,
                )
                scores = support_scores(
                    predicted,
                    batch["gold_supporting_facts"][index],
                )
                for name, value in zip(totals, scores):
                    totals[name] += value
                predictions.append(
                    {
                        "id": example_id,
                        "supporting_facts": predicted,
                        "visible_sentences": int(node_mask[index].sum().item()),
                        "candidate_sentences": len(batch["node_metadata"][index]),
                    }
                )
                examples_seen += 1

    if not examples_seen:
        raise ValueError("The validation split is empty.")
    metrics = {name: value / examples_seen for name, value in totals.items()}
    metrics["examples"] = examples_seen
    return metrics, predictions


def train_transformer_support_model(
    model,
    train_loader,
    validation_loader,
    device,
    config,
    xla_model=None,
    rank=0,
    train_sampler=None,
):
    """Train the BERT-only model and save its best support-fact checkpoint."""
    checkpoint_dir = Path(config["checkpoint_dir"])
    run_dir = Path(config["run_dir"])
    is_master = rank == 0
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        total_loss = 0.0
        batches = 0
        for batch in tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            disable=not is_master,
        ):
            batch = _device_batch(batch, device)
            outputs = model(**{key: batch[key] for key in MODEL_INPUTS})
            loss = _support_loss(
                outputs["support_logits"],
                batch["support_labels"],
                batch["node_mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            if xla_model is None:
                optimizer.step()
            else:
                xla_model.optimizer_step(optimizer)
            total_loss += float(loss.detach().cpu())
            batches += 1

        if not batches:
            raise ValueError("The training split is empty.")
        train_loss = total_loss / batches
        if xla_model is not None:
            train_loss = xla_model.mesh_reduce(
                f"transformer-train-loss-{epoch}",
                train_loss,
                lambda values: sum(values) / len(values),
            )

        control = None
        if is_master:
            validation_metrics, predictions = evaluate_support_model(
                model, validation_loader, device, config
            )
            record = {
                "epoch": epoch,
                "train_support_loss": train_loss,
                "validation": validation_metrics,
            }
            history.append(record)
            with (run_dir / "training_history.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=2, ensure_ascii=False)
                file.write("\n")

            print(
                f"Epoch {epoch}: support EM={validation_metrics['support_em']:.4f}, "
                f"support F1={validation_metrics['support_f1']:.4f}"
            )
            if validation_metrics["support_f1"] > best_validation_f1:
                best_validation_f1 = validation_metrics["support_f1"]
                epochs_without_improvement = 0
                with (run_dir / "validation_predictions.jsonl").open(
                    "w", encoding="utf-8"
                ) as file:
                    for prediction in predictions:
                        file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                model_state = {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state": model_state,
                        "model_name": config["model_name"],
                        "model_config": dict(config),
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                    },
                    checkpoint_dir / "best_transformer_support.pt",
                )
                print(
                    "Saved best transformer support checkpoint to "
                    f"{checkpoint_dir / 'best_transformer_support.pt'}"
                )
            else:
                epochs_without_improvement += 1

            control = {
                "best_validation_f1": best_validation_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "should_stop": (
                    epochs_without_improvement
                    >= config["early_stopping_patience"]
                ),
            }

        if xla_model is not None:
            control = xla_model.mesh_reduce(
                f"transformer-epoch-control-{epoch}",
                control,
                lambda values: values[0],
            )
            best_validation_f1 = control["best_validation_f1"]
            epochs_without_improvement = control["epochs_without_improvement"]
        if control["should_stop"]:
            if is_master:
                print("Stopping because validation support F1 did not improve.")
            break
