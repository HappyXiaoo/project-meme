from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from app.services.classifier import get_meme_feature_names


class BertMemeFusionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        fusion_hidden_size: int = 128,
        dropout: float = 0.1,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size
        self.meme_feature_dim = len(get_meme_feature_names())
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(bert_hidden + self.meme_feature_dim, fusion_hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(fusion_hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        meme_features=None,
        labels=None,
        **kwargs,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        if meme_features is None:
            raise ValueError("meme_features is required for BertMemeFusionModel")

        if meme_features.dtype != cls_output.dtype:
            meme_features = meme_features.to(dtype=cls_output.dtype)

        fused = torch.cat([cls_output, meme_features], dim=-1)
        fused = self.activation(self.fusion(fused))
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
