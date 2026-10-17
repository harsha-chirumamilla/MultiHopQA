"""Losses for answer type, extractive spans, and sentence support labels."""

import torch
from torch.nn import functional as F


def calculate_qa_loss(outputs, batch, support_loss_weight, span_loss_weight):
    """Combine the three training objectives for one batch."""
    answer_type_loss = F.cross_entropy(
        outputs["answer_type_logits"],
        batch["answer_type_labels"],
    )

    node_loss = F.binary_cross_entropy_with_logits(
        outputs["support_logits"],
        batch["support_labels"],
        reduction="none",
    )
    valid_nodes = batch["node_mask"]
    positive_nodes = valid_nodes & (batch["support_labels"] > 0.5)
    negative_nodes = valid_nodes & ~positive_nodes

    if positive_nodes.any():
        positive_loss = node_loss[positive_nodes].mean()
    else:
        positive_loss = node_loss.sum() * 0.0

    if negative_nodes.any():
        negative_loss = node_loss[negative_nodes].mean()
    else:
        negative_loss = node_loss.sum() * 0.0

    support_loss = (positive_loss + negative_loss) / 2.0

    span_losses = []
    batch_size, _, sequence_length = outputs["start_logits"].shape
    for graph_index in range(batch_size):
        if not batch["span_available"][graph_index]:
            continue

        node_count = int(batch["node_mask"][graph_index].sum().item())
        start_scores = outputs["start_logits"][graph_index, :node_count].reshape(-1)
        end_scores = outputs["end_logits"][graph_index, :node_count].reshape(-1)

        start_target = (
            batch["span_node"][graph_index] * sequence_length
            + batch["span_start"][graph_index]
        )
        end_target = (
            batch["span_node"][graph_index] * sequence_length
            + batch["span_end"][graph_index]
        )

        start_loss = F.cross_entropy(start_scores.unsqueeze(0), start_target.unsqueeze(0))
        end_loss = F.cross_entropy(end_scores.unsqueeze(0), end_target.unsqueeze(0))
        span_losses.append((start_loss + end_loss) / 2.0)

    if span_losses:
        span_loss = torch.stack(span_losses).mean()
    else:
        span_loss = outputs["start_logits"].sum() * 0.0

    total_loss = (
        answer_type_loss
        + support_loss_weight * support_loss
        + span_loss_weight * span_loss
    )

    return {
        "total": total_loss,
        "answer_type": answer_type_loss,
        "support": support_loss,
        "span": span_loss,
        "span_count": len(span_losses),
    }
