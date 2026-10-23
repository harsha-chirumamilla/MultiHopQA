"""Joint answer and evidence training for the controlled HotpotQA readers."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from multihopqa.evaluation.answer_metrics import answer_scores
from multihopqa.evaluation.joint_metrics import joint_scores
from multihopqa.evaluation.support_metrics import support_scores


MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "context_mask",
    "token_sentence_ids",
    "window_mask",
    "node_mask",
    "adjacency",
)


def _move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _calculate_loss(outputs, batch, config):
    answer_type_loss = F.cross_entropy(
        outputs["answer_type_logits"], batch["answer_type_labels"]
    )

    node_losses = F.binary_cross_entropy_with_logits(
        outputs["support_logits"], batch["support_labels"], reduction="none"
    )
    valid_nodes = batch["node_mask"]
    positive_nodes = valid_nodes & (batch["support_labels"] > 0.5)
    negative_nodes = valid_nodes & ~positive_nodes
    positive_loss = (node_losses * positive_nodes).sum() / positive_nodes.sum().clamp(min=1)
    negative_loss = (node_losses * negative_nodes).sum() / negative_nodes.sum().clamp(min=1)
    support_loss = (positive_loss + negative_loss) / 2.0

    batch_size, window_count, sequence_length = outputs["start_logits"].shape
    start_scores = outputs["start_logits"].reshape(batch_size, -1)
    end_scores = outputs["end_logits"].reshape(batch_size, -1)
    start_targets = batch["span_window"] * sequence_length + batch["span_start"]
    end_targets = batch["span_window"] * sequence_length + batch["span_end"]
    start_losses = F.cross_entropy(start_scores, start_targets, reduction="none")
    end_losses = F.cross_entropy(end_scores, end_targets, reduction="none")
    available = batch["span_available"].to(start_losses.dtype)
    span_loss = (((start_losses + end_losses) / 2.0) * available).sum() / available.sum().clamp(min=1.0)

    total_loss = (
        config.get("answer_type_loss_weight", 1.0) * answer_type_loss
        + config.get("support_loss_weight", 1.0) * support_loss
        + config.get("span_loss_weight", 1.0) * span_loss
    )
    return {
        "total": total_loss,
        "answer_type": answer_type_loss,
        "support": support_loss,
        "span": span_loss,
    }


def _decode_span(start_logits, end_logits, context_mask, sentence_ids, input_ids,
                 tokenizer, max_answer_length, allowed_sentences):
    """Select a valid span within one predicted supporting sentence."""
    window_count, sequence_length = start_logits.shape
    positions = torch.arange(sequence_length, device=start_logits.device)
    starts = positions.view(1, sequence_length, 1)
    ends = positions.view(1, 1, sequence_length)
    ordered = ends >= starts
    short_enough = (ends - starts) < max_answer_length
    same_sentence = sentence_ids.unsqueeze(2) == sentence_ids.unsqueeze(1)
    valid = (
        context_mask.unsqueeze(2)
        & context_mask.unsqueeze(1)
        & (sentence_ids.unsqueeze(2) >= 0)
        & same_sentence
        & ordered
        & short_enough
    )
    if allowed_sentences:
        allowed = torch.zeros_like(sentence_ids, dtype=torch.bool)
        for sentence_id in allowed_sentences:
            allowed |= sentence_ids == sentence_id
        valid &= allowed.unsqueeze(2) & allowed.unsqueeze(1)

    scores = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)
    scores = scores.masked_fill(~valid, -10000.0)
    best_flat = int(scores.reshape(-1).argmax().item())
    pair_count = sequence_length * sequence_length
    window_index = best_flat // pair_count
    pair_index = best_flat % pair_count
    start_index, end_index = divmod(pair_index, sequence_length)
    if not valid[window_index, start_index, end_index]:
        return ""
    return tokenizer.decode(
        input_ids[window_index, start_index : end_index + 1].tolist(),
        skip_special_tokens=True,
    ).strip()


def _select_support_indices(logits, node_mask, top_k):
    available = int(node_mask.sum().item())
    count = min(int(top_k), available)
    if count == 0:
        return []
    return logits.masked_fill(~node_mask, -10000.0).topk(count).indices.tolist()


def evaluate_reader(model, data_loader, device, tokenizer, config):
    model.eval()
    names = ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")
    totals = {name: 0.0 for name in names}
    predictions = []
    seen = 0
    span_misses = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validate"):
            batch = _move_batch(batch, device)
            outputs = model(**{key: batch[key] for key in MODEL_INPUTS})
            if device.type == "xla":
                outputs = {key: value.cpu() for key, value in outputs.items()}
                batch = {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }

            predicted_types = outputs["answer_type_logits"].argmax(dim=-1)
            span_misses += int(
                ((batch["answer_type_labels"] == 2) & ~batch["span_available"]).sum().item()
            )
            for example_index, example_id in enumerate(batch["example_ids"]):
                metadata = batch["node_metadata"][example_index]
                node_count = len(metadata)
                support_logits = outputs["support_logits"][example_index, :node_count]
                node_mask = batch["node_mask"][example_index, :node_count]
                allowed_sentences = _select_support_indices(
                    support_logits, node_mask, config.get("support_top_k", 2)
                )
                predicted_support = [metadata[index] for index in allowed_sentences]

                answer_type = int(predicted_types[example_index].item())
                if answer_type == 0:
                    predicted_answer = "yes"
                elif answer_type == 1:
                    predicted_answer = "no"
                else:
                    valid_windows = int(batch["window_mask"][example_index].sum().item())
                    predicted_answer = _decode_span(
                        outputs["start_logits"][example_index, :valid_windows],
                        outputs["end_logits"][example_index, :valid_windows],
                        batch["context_mask"][example_index, :valid_windows],
                        batch["token_sentence_ids"][example_index, :valid_windows],
                        batch["input_ids"][example_index, :valid_windows],
                        tokenizer,
                        int(config.get("max_answer_length", 30)),
                        allowed_sentences,
                    )

                answer_result = answer_scores(
                    predicted_answer, batch["gold_answers"][example_index]
                )
                support_result = support_scores(
                    predicted_support, batch["gold_supporting_facts"][example_index]
                )
                joint_em, joint_f1 = joint_scores(answer_result, support_result)
                totals["answer_em"] += answer_result[0]
                totals["answer_f1"] += answer_result[3]
                totals["support_em"] += support_result[0]
                totals["support_f1"] += support_result[3]
                totals["joint_em"] += joint_em
                totals["joint_f1"] += joint_f1
                predictions.append(
                    {"id": example_id, "answer": predicted_answer, "supporting_facts": predicted_support}
                )
                seen += 1

    if not seen:
        raise ValueError("The validation split is empty.")
    metrics = {name: value / seen for name, value in totals.items()}
    metrics["examples"] = seen
    metrics["span_alignment_misses"] = span_misses
    return metrics, predictions


def _average_metrics(values):
    count = len(values)
    return {key: sum(item[key] for item in values) / count for key in values[0]}


def train_reader(model, train_loader, validation_loader, device, tokenizer, config,
                 xla_model=None, rank=0, train_sampler=None):
    is_master = rank == 0
    checkpoint_dir = Path(config["checkpoint_dir"])
    run_dir = Path(config["run_dir"])
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    total_steps = max(1, len(train_loader) * int(config["epochs"]))
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))

    def learning_rate_scale(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    best_joint_f1 = -1.0
    patience_count = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        sums = {"total": 0.0, "answer_type": 0.0, "support": 0.0, "span": 0.0}
        batches = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", disable=not is_master):
            batch = _move_batch(batch, device)
            outputs = model(**{key: batch[key] for key in MODEL_INPUTS})
            losses = _calculate_loss(outputs, batch, config)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if xla_model is None:
                optimizer.step()
            else:
                xla_model.optimizer_step(optimizer)
            scheduler.step()
            for name in sums:
                sums[name] += float(losses[name].detach().cpu())
            batches += 1

        if not batches:
            raise ValueError("The training split is empty.")
        train_metrics = {key: value / batches for key, value in sums.items()}
        if xla_model is not None:
            train_metrics = xla_model.mesh_reduce(
                f"reader-train-metrics-{epoch}", train_metrics, _average_metrics
            )

        control = None
        if is_master:
            validation_metrics, predictions = evaluate_reader(
                model, validation_loader, device, tokenizer, config
            )
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
            with (run_dir / "training_history.json").open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2, ensure_ascii=False)
                file.write("\n")
            print(
                f"Epoch {epoch}: answer EM={validation_metrics['answer_em']:.4f}, "
                f"answer F1={validation_metrics['answer_f1']:.4f}, "
                f"support F1={validation_metrics['support_f1']:.4f}, "
                f"joint F1={validation_metrics['joint_f1']:.4f}; "
                f"unmatched span labels={validation_metrics['span_alignment_misses']}"
            )

            if validation_metrics["joint_f1"] > best_joint_f1:
                best_joint_f1 = validation_metrics["joint_f1"]
                patience_count = 0
                with (run_dir / "validation_predictions.jsonl").open(
                    "w", encoding="utf-8"
                ) as file:
                    for prediction in predictions:
                        file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                torch.save(
                    {
                        "model_state": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "model_name": config["model_name"],
                        "use_graph": config["use_graph"],
                        "model_config": dict(config),
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                    },
                    checkpoint_dir / "best_reader.pt",
                )
            else:
                patience_count += 1
            control = {
                "best_joint_f1": best_joint_f1,
                "patience_count": patience_count,
                "stop": patience_count >= config.get("early_stopping_patience", 2),
            }

        if xla_model is not None:
            control = xla_model.mesh_reduce(
                f"reader-control-{epoch}", control, lambda values: values[0]
            )
        if control["stop"]:
            if is_master:
                print("Stopping because validation joint F1 did not improve.")
            break
    return history
