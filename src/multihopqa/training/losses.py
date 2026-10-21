"""Losses for answer type, extractive spans, and sentence support labels."""

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

    positive_loss = (node_loss * positive_nodes).sum() / positive_nodes.sum().clamp(
        min=1
    )
    negative_loss = (node_loss * negative_nodes).sum() / negative_nodes.sum().clamp(
        min=1
    )

    support_loss = (positive_loss + negative_loss) / 2.0

    _, _, sequence_length = outputs["start_logits"].shape
    start_scores = outputs["start_logits"].flatten(start_dim=1)
    end_scores = outputs["end_logits"].flatten(start_dim=1)
    start_targets = batch["span_node"] * sequence_length + batch["span_start"]
    end_targets = batch["span_node"] * sequence_length + batch["span_end"]
    start_losses = F.cross_entropy(start_scores, start_targets, reduction="none")
    end_losses = F.cross_entropy(end_scores, end_targets, reduction="none")
    available_spans = batch["span_available"].to(outputs["start_logits"].dtype)
    span_loss = (
        ((start_losses + end_losses) / 2.0) * available_spans
    ).sum() / available_spans.sum().clamp(min=1.0)

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
    }
