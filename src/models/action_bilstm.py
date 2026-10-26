"""
Module: action_bilstm.py
Purpose: BiLSTM with Multi-Head Self-Attention for Skeleton-based Action Recognition.
         Inspired by and adapted from the MMASD BiLSTM Baseline architecture.
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ActionBiLSTMAttentionClassifier(nn.Module):
    """
    BiLSTM followed by Masked Multi-Head Self-Attention and Temporal Pooling
    for human motion action classification from 2D skeleton sequences.
    """

    def __init__(
        self,
        input_size: int = 24,  # 12 joints * 2 (x, y) coordinates
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 15,  # 15 Penn Action classes
        attention_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last_valid"}:
            raise ValueError("pooling must be 'mean' or 'last_valid'")

        directions = 2 if bidirectional else 1
        output_size = directions * hidden_size
        if output_size % attention_heads != 0:
            raise ValueError(
                f"LSTM output size {output_size} must be divisible by attention_heads={attention_heads}"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.pooling = pooling
        self.num_classes = num_classes

        # 1. BiLSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Regularization & Classification Head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(
        self,
        inputs: Tensor,
        lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            inputs: [Batch, Time, input_size] or [Batch, Time, Joints, 2]
            lengths: [Batch] sequence lengths before padding
            padding_mask: [Batch, Time] boolean mask where True indicates padded/invalid frame
            return_attention: whether to return attention weights
        """
        if inputs.ndim == 4:
            inputs = inputs.flatten(start_dim=2)

        batch, time, feat_dim = inputs.shape
        if feat_dim != self.input_size:
            raise ValueError(f"Expected feature dim {self.input_size}, got {feat_dim}")

        if lengths is None:
            if padding_mask is None:
                lengths = torch.full((batch,), time, dtype=torch.long, device=inputs.device)
            else:
                lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)

        lengths = lengths.to(device=inputs.device, dtype=torch.long).clamp(min=1, max=time)

        if padding_mask is None:
            padding_mask = torch.arange(time, device=inputs.device).unsqueeze(
                0
            ) >= lengths.unsqueeze(1)
        else:
            padding_mask = padding_mask.to(device=inputs.device, dtype=torch.bool)

        # 1. Pass through BiLSTM using packed sequence
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=time)

        # 2. Multi-Head Self-Attention over sequence time
        attended, attention_weights = self.attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
        )

        # 3. Masked Temporal Pooling
        valid = (~padding_mask).unsqueeze(-1).to(attended.dtype)
        if self.pooling == "mean":
            pooled = (attended * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            last_indices = lengths - 1
            pooled = attended[torch.arange(batch, device=inputs.device), last_indices]

        # 4. Classification Logits
        logits = self.classifier(self.dropout(pooled))

        if return_attention:
            return logits, attention_weights
        return logits
