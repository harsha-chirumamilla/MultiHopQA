"""Frozen BERT text encoding followed by a two-layer sentence GAT."""

import torch
from torch import nn

from multihopqa.models.gat_layer import GraphAttentionLayer


class BasicGATForQuestionAnswering(nn.Module):
    """Predict answer type, answer span, and supporting sentences."""

    def __init__(
        self,
        encoder,
        graph_hidden_size=128,
        attention_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_size = encoder.config.hidden_size

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.first_gat = GraphAttentionLayer(
            self.encoder_size,
            graph_hidden_size,
            attention_heads,
            dropout,
        )
        self.second_gat = GraphAttentionLayer(
            graph_hidden_size,
            graph_hidden_size,
            1,
            dropout,
        )

        self.support_head = nn.Linear(graph_hidden_size, 1)
        self.answer_type_head = nn.Linear(graph_hidden_size, 3)
        self.graph_to_token = nn.Linear(graph_hidden_size, self.encoder_size)
        self.start_head = nn.Linear(self.encoder_size, 1)
        self.end_head = nn.Linear(self.encoder_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        context_mask,
        node_mask,
        adjacency,
    ):
        batch_size, node_count, sequence_length = input_ids.shape
        flat_node_mask = node_mask.reshape(-1)

        flat_input_ids = input_ids.reshape(-1, sequence_length)
        flat_attention_mask = attention_mask.reshape(-1, sequence_length)
        flat_token_type_ids = token_type_ids.reshape(-1, sequence_length)

        # The BERT encoder stays frozen in the first experiment.
        self.encoder.eval()
        with torch.no_grad():
            encoded = self.encoder(
                input_ids=flat_input_ids[flat_node_mask],
                attention_mask=flat_attention_mask[flat_node_mask],
                token_type_ids=flat_token_type_ids[flat_node_mask],
            )

        valid_token_states = encoded.last_hidden_state
        token_states = valid_token_states.new_zeros(
            (
                batch_size * node_count,
                sequence_length,
                self.encoder_size,
            )
        )
        token_states[flat_node_mask] = valid_token_states
        token_states = token_states.view(
            batch_size,
            node_count,
            sequence_length,
            self.encoder_size,
        )

        sentence_features = token_states[:, :, 0, :]
        graph_nodes = self.first_gat(sentence_features, adjacency, node_mask)
        graph_nodes = self.second_gat(graph_nodes, adjacency, node_mask)

        node_weights = node_mask.unsqueeze(-1).to(graph_nodes.dtype)
        graph_summary = (graph_nodes * node_weights).sum(dim=1)
        graph_summary = graph_summary / node_weights.sum(dim=1).clamp(min=1.0)

        answer_type_logits = self.answer_type_head(graph_summary)
        support_logits = self.support_head(graph_nodes).squeeze(-1)

        graph_context = self.graph_to_token(graph_nodes).unsqueeze(2)
        token_features = token_states + graph_context
        start_logits = self.start_head(token_features).squeeze(-1)
        end_logits = self.end_head(token_features).squeeze(-1)

        valid_context_tokens = context_mask & node_mask.unsqueeze(-1)
        start_logits = start_logits.masked_fill(~valid_context_tokens, -10000.0)
        end_logits = end_logits.masked_fill(~valid_context_tokens, -10000.0)

        return {
            "answer_type_logits": answer_type_logits,
            "support_logits": support_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
