"""Fine-tuned BERT reader with an optional sentence graph reasoning layer."""

import torch
from torch import nn

from multihopqa.models.gat_layer import GraphAttentionLayer


class HotpotQAReader(nn.Module):
    """Jointly predict an answer and supporting facts from overlapping windows.

    Setting ``use_graph=False`` gives the controlled BERT reader baseline.
    Setting it to true adds sentence-level graph propagation while keeping the
    encoder, heads, inputs, labels, and losses identical.
    """

    def __init__(
        self,
        encoder,
        use_graph=False,
        graph_hidden_size=128,
        attention_heads=4,
        dropout=0.1,
        answer_support_bias=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_size = encoder.config.hidden_size
        self.use_graph = use_graph
        self.answer_support_bias = answer_support_bias
        self.dropout = nn.Dropout(dropout)

        if use_graph:
            self.first_gat = GraphAttentionLayer(
                self.encoder_size, graph_hidden_size, attention_heads, dropout
            )
            self.second_gat = GraphAttentionLayer(
                graph_hidden_size, graph_hidden_size, 1, dropout
            )
            sentence_size = graph_hidden_size
            self.graph_to_token = nn.Linear(graph_hidden_size, self.encoder_size)
        else:
            sentence_size = self.encoder_size
            self.first_gat = None
            self.second_gat = None
            self.graph_to_token = None

        self.support_head = nn.Linear(sentence_size, 1)
        self.answer_type_head = nn.Linear(sentence_size, 3)
        self.start_head = nn.Linear(self.encoder_size, 1)
        self.end_head = nn.Linear(self.encoder_size, 1)

    def _pool_sentences(self, token_states, token_sentence_ids, node_count):
        """Mean-pool every visible sentence across all overlapping windows."""
        batch_size, window_count, sequence_length, hidden_size = token_states.shape
        sentence_states = []
        sentence_counts = []
        for batch_index in range(batch_size):
            token_ids = token_sentence_ids[batch_index].reshape(-1)
            flat_states = token_states[batch_index].reshape(-1, hidden_size)
            valid = (token_ids >= 0) & (token_ids < node_count)
            ids = token_ids.clamp(min=0, max=node_count - 1)
            weights = valid.to(flat_states.dtype)
            sums = flat_states.new_zeros((node_count, hidden_size)).index_add(
                0, ids, flat_states * weights.unsqueeze(-1)
            )
            counts = flat_states.new_zeros((node_count, 1)).index_add(
                0, ids, weights.unsqueeze(-1)
            )
            sentence_states.append(sums / counts.clamp(min=1.0))
            sentence_counts.append(counts.squeeze(-1))
        return torch.stack(sentence_states), torch.stack(sentence_counts)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        context_mask,
        token_sentence_ids,
        window_mask,
        node_mask,
        adjacency,
    ):
        batch_size, window_count, sequence_length = input_ids.shape
        encoded = self.encoder(
            input_ids=input_ids.reshape(-1, sequence_length),
            attention_mask=attention_mask.reshape(-1, sequence_length),
            token_type_ids=token_type_ids.reshape(-1, sequence_length),
        )
        token_states = encoded.last_hidden_state.view(
            batch_size, window_count, sequence_length, self.encoder_size
        )

        node_count = node_mask.shape[1]
        sentence_features, sentence_token_counts = self._pool_sentences(
            token_states, token_sentence_ids, node_count
        )
        if self.use_graph:
            sentence_features = self.first_gat(sentence_features, adjacency, node_mask)
            sentence_features = self.second_gat(sentence_features, adjacency, node_mask)

        support_logits = self.support_head(self.dropout(sentence_features)).squeeze(-1)
        support_logits = support_logits.masked_fill(~node_mask, -10000.0)

        valid_nodes = node_mask & (sentence_token_counts > 0)
        summary_weights = valid_nodes.unsqueeze(-1).to(sentence_features.dtype)
        question_summary = (sentence_features * summary_weights).sum(dim=1)
        question_summary = question_summary / summary_weights.sum(dim=1).clamp(min=1.0)
        answer_type_logits = self.answer_type_head(self.dropout(question_summary))

        token_features = token_states
        if self.use_graph:
            safe_sentence_ids = token_sentence_ids.clamp(min=0, max=max(node_count - 1, 0))
            graph_states = self.graph_to_token(sentence_features)
            token_graph_context = torch.gather(
                graph_states.unsqueeze(1).expand(-1, window_count, -1, -1),
                2,
                safe_sentence_ids.unsqueeze(-1).expand(-1, -1, -1, self.encoder_size),
            )
            token_graph_context = token_graph_context * (
                token_sentence_ids >= 0
            ).unsqueeze(-1)
            token_features = token_features + token_graph_context

        token_features = self.dropout(token_features)
        start_logits = self.start_head(token_features).squeeze(-1)
        end_logits = self.end_head(token_features).squeeze(-1)

        safe_sentence_ids = token_sentence_ids.clamp(min=0, max=max(node_count - 1, 0))
        token_support_logits = torch.gather(
            support_logits.unsqueeze(1).expand(-1, window_count, -1),
            2,
            safe_sentence_ids,
        )
        support_log_probability = torch.nn.functional.logsigmoid(token_support_logits)
        support_log_probability = support_log_probability * (token_sentence_ids >= 0)
        start_logits = start_logits + self.answer_support_bias * support_log_probability
        end_logits = end_logits + self.answer_support_bias * support_log_probability

        valid_answer_tokens = context_mask & (token_sentence_ids >= 0)
        start_logits = start_logits.masked_fill(~valid_answer_tokens, -10000.0)
        end_logits = end_logits.masked_fill(~valid_answer_tokens, -10000.0)
        start_logits = start_logits.masked_fill(~window_mask.unsqueeze(-1), -10000.0)
        end_logits = end_logits.masked_fill(~window_mask.unsqueeze(-1), -10000.0)

        return {
            "answer_type_logits": answer_type_logits,
            "support_logits": support_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
