"""BERT-only global-context model for HotpotQA support-fact prediction."""

import torch
from torch import nn


class TransformerSupportModel(nn.Module):
    """Encode the full question context once and score its sentence evidence."""

    def __init__(self, encoder, freeze_encoder=True, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.encoder_size = encoder.config.hidden_size
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.support_head = nn.Linear(self.encoder_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        sentence_token_mask,
        node_mask,
    ):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                token_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).last_hidden_state
        else:
            token_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

        sentence_weights = sentence_token_mask.to(token_states.dtype)
        sentence_sums = torch.bmm(sentence_weights, token_states)
        sentence_counts = sentence_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sentence_features = sentence_sums / sentence_counts
        support_logits = self.support_head(self.dropout(sentence_features)).squeeze(-1)
        support_logits = support_logits.masked_fill(~node_mask, -10000.0)
        return {"support_logits": support_logits}
