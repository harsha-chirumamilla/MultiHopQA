"""Training and validation loops for the first Basic GAT experiment."""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.answer_metrics import answer_scores
from multihopqa.evaluation.joint_metrics import joint_scores
from multihopqa.evaluation.support_metrics import support_scores
from multihopqa.training.losses import calculate_qa_loss


MODEL_INPUTS = [
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "context_mask",
    "node_mask",
    "adjacency",
]


def move_batch_to_device(batch, device):
    """Move tensor values to the selected device and keep text metadata on CPU."""
    device_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device)
        else:
            device_batch[key] = value
    return device_batch


def get_model_inputs(batch):
    """Select the tensors the model needs for its forward pass."""
    return {key: batch[key] for key in MODEL_INPUTS}


def train_one_epoch(model, data_loader, optimizer, device, config, epoch_number):
    """Run one full pass through the training examples."""
    model.train()
    totals = {"total": 0.0, "answer_type": 0.0, "support": 0.0, "span": 0.0}
    batches_seen = 0
    missing_span_labels = 0

    progress = tqdm(data_loader, desc=f"Train epoch {epoch_number}")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        outputs = model(**get_model_inputs(batch))
        losses = calculate_qa_loss(
            outputs,
            batch,
            config["support_loss_weight"],
            config["span_loss_weight"],
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for name in totals:
            totals[name] += losses[name].item()
        batches_seen += 1
        missing_span_labels += int(
            (
                (batch["answer_type_labels"] == 2)
                & ~batch["span_available"]
            ).sum().item()
        )

        progress.set_postfix(
            loss=f"{losses['total'].item():.3f}",
            answer=f"{losses['answer_type'].item():.3f}",
            support=f"{losses['support'].item():.3f}",
            span=f"{losses['span'].item():.3f}",
        )

    if batches_seen == 0:
        raise ValueError("The training split is empty.")

    averages = {name: value / batches_seen for name, value in totals.items()}
    averages["span_alignment_misses"] = missing_span_labels
    return averages


def decode_answer_span(
    start_logits,
    end_logits,
    context_mask,
    input_ids,
    max_answer_length,
    tokenizer,
):
    """Choose the highest-scoring valid span from one question graph."""
    node_count, sequence_length = start_logits.shape
    positions = torch.arange(sequence_length, device=start_logits.device)

    valid_order = positions.unsqueeze(0) >= positions.unsqueeze(1)
    valid_length = (
        positions.unsqueeze(0) - positions.unsqueeze(1)
    ) < max_answer_length
    valid_tokens = context_mask.unsqueeze(2) & context_mask.unsqueeze(1)
    valid_spans = valid_tokens & valid_order.unsqueeze(0) & valid_length.unsqueeze(0)

    span_scores = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)
    span_scores = span_scores.masked_fill(~valid_spans, -10000.0)

    best_index = int(span_scores.reshape(-1).argmax().item())
    span_size = sequence_length * sequence_length
    node_index = best_index // span_size
    position_index = best_index % span_size
    start_index = position_index // sequence_length
    end_index = position_index % sequence_length

    if node_index >= node_count or not valid_spans[node_index, start_index, end_index]:
        return ""

    token_ids = input_ids[node_index, start_index : end_index + 1].tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def evaluate_model(model, data_loader, device, tokenizer, config):
    """Predict answers and support facts for the full validation split."""
    model.eval()
    metric_totals = {
        "answer_em": 0.0,
        "answer_f1": 0.0,
        "support_em": 0.0,
        "support_f1": 0.0,
        "joint_em": 0.0,
        "joint_f1": 0.0,
    }
    predictions = []
    examples_seen = 0
    missing_span_labels = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validate"):
            batch = move_batch_to_device(batch, device)
            outputs = model(**get_model_inputs(batch))
            answer_types = outputs["answer_type_logits"].argmax(dim=-1)
            node_counts = batch["node_mask"].sum(dim=1)

            missing_span_labels += int(
                (
                    (batch["answer_type_labels"] == 2)
                    & ~batch["span_available"]
                ).sum().item()
            )

            for graph_index in range(len(batch["example_ids"])):
                node_count = int(node_counts[graph_index].item())
                answer_type = int(answer_types[graph_index].item())

                if answer_type == 0:
                    predicted_answer = "yes"
                elif answer_type == 1:
                    predicted_answer = "no"
                else:
                    predicted_answer = decode_answer_span(
                        outputs["start_logits"][graph_index, :node_count],
                        outputs["end_logits"][graph_index, :node_count],
                        batch["context_mask"][graph_index, :node_count],
                        batch["input_ids"][graph_index, :node_count],
                        config["max_answer_length"],
                        tokenizer,
                    )

                support_probabilities = torch.sigmoid(
                    outputs["support_logits"][graph_index, :node_count]
                )
                predicted_support = []
                for node_index, probability in enumerate(support_probabilities):
                    if probability.item() >= 0.5:
                        predicted_support.append(
                            batch["node_metadata"][graph_index][node_index]
                        )

                answer_score = answer_scores(
                    predicted_answer,
                    batch["gold_answers"][graph_index],
                )
                support_score = support_scores(
                    predicted_support,
                    batch["gold_supporting_facts"][graph_index],
                )
                joint_em, joint_f1 = joint_scores(answer_score, support_score)

                metric_totals["answer_em"] += answer_score[0]
                metric_totals["answer_f1"] += answer_score[3]
                metric_totals["support_em"] += support_score[0]
                metric_totals["support_f1"] += support_score[3]
                metric_totals["joint_em"] += joint_em
                metric_totals["joint_f1"] += joint_f1
                examples_seen += 1

                predictions.append(
                    {
                        "id": batch["example_ids"][graph_index],
                        "answer": predicted_answer,
                        "supporting_facts": predicted_support,
                    }
                )

    if examples_seen == 0:
        raise ValueError("The validation split is empty.")

    metrics = {
        name: value / examples_seen
        for name, value in metric_totals.items()
    }
    metrics["examples"] = examples_seen
    metrics["span_alignment_misses"] = missing_span_labels
    return metrics, predictions


def save_json(file_path, value):
    """Write a readable JSON file."""
    with open(file_path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def save_predictions(file_path, predictions):
    """Write one validation prediction per line."""
    with open(file_path, "w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def train_model(model, train_loader, validation_loader, device, tokenizer, config):
    """Train, validate, and save the best checkpoint."""
    checkpoint_dir = Path(config["checkpoint_dir"])
    run_dir = Path(config["run_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
    )

    if isinstance(model, torch.nn.DataParallel):
        checkpoint_model = model.module
    else:
        checkpoint_model = model

    best_joint_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch_number in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            epoch_number,
        )
        validation_metrics, predictions = evaluate_model(
            model,
            validation_loader,
            device,
            tokenizer,
            config,
        )

        epoch_record = {
            "epoch": epoch_number,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        save_json(run_dir / "training_history.json", history)

        print(
            f"Epoch {epoch_number}: "
            f"validation answer F1={validation_metrics['answer_f1']:.4f}, "
            f"support F1={validation_metrics['support_f1']:.4f}, "
            f"joint F1={validation_metrics['joint_f1']:.4f}"
        )
        print(
            f"Span labels not found in context: "
            f"train={train_metrics['span_alignment_misses']}, "
            f"validation={validation_metrics['span_alignment_misses']}"
        )

        if validation_metrics["joint_f1"] > best_joint_f1:
            best_joint_f1 = validation_metrics["joint_f1"]
            epochs_without_improvement = 0
            save_predictions(run_dir / "validation_predictions.jsonl", predictions)
            torch.save(
                {
                    "model_state": checkpoint_model.state_dict(),
                    "model_name": config["model_name"],
                    "graph_hidden_size": config["graph_hidden_size"],
                    "attention_heads": config["attention_heads"],
                    "model_config": dict(config),
                    "epoch": epoch_number,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_dir / "best_basic_gat.pt",
            )
            print(f"Saved best checkpoint to {checkpoint_dir / 'best_basic_gat.pt'}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print("Stopping because validation joint F1 did not improve.")
            break

    return history
