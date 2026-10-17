"""A small dense-graph multi-head attention layer."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    """Update each sentence using attention over its connected sentences."""

    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()

        if output_size % head_count != 0:
            raise ValueError("output_size must divide evenly across attention heads.")

        self.output_size = output_size
        self.head_count = head_count
        self.head_size = output_size // head_count
        self.dropout = dropout

        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.source_attention = nn.Parameter(
            torch.empty(head_count, self.head_size)
        )
        self.target_attention = nn.Parameter(
            torch.empty(head_count, self.head_size)
        )

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, node_features, adjacency, node_mask):
        batch_size, node_count, _ = node_features.shape
        node_features = F.dropout(
            node_features,
            p=self.dropout,
            training=self.training,
        )

        projected = self.linear(node_features)
        projected = projected.view(
            batch_size,
            node_count,
            self.head_count,
            self.head_size,
        )

        source_scores = (
            projected * self.source_attention.view(1, 1, self.head_count, self.head_size)
        ).sum(dim=-1)
        target_scores = (
            projected * self.target_attention.view(1, 1, self.head_count, self.head_size)
        ).sum(dim=-1)

        scores = source_scores.unsqueeze(2) + target_scores.unsqueeze(1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        scores = scores.masked_fill(~adjacency.unsqueeze(-1), -10000.0)

        attention = torch.softmax(scores, dim=2)
        attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )

        head_outputs = []
        for head_index in range(self.head_count):
            head_attention = attention[:, :, :, head_index]
            head_features = projected[:, :, head_index, :]
            head_outputs.append(torch.bmm(head_attention, head_features))

        updated_nodes = torch.cat(head_outputs, dim=-1)
        updated_nodes = F.elu(updated_nodes)
        updated_nodes = updated_nodes * node_mask.unsqueeze(-1)
        return updated_nodes
