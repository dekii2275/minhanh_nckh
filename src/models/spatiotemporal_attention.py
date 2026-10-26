"""
Module: spatiotemporal_attention.py
Purpose: Spatio-Temporal Dual Attention Network for Skeleton Action Recognition.
         Applies Spatial Multi-Head Self-Attention across the 12 body joints at each frame,
         followed by a Bidirectional LSTM and Temporal Self-Attention across video time.
"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpatioTemporalAttentionClassifier(nn.Module):
    """
    Dual Spatial-Temporal Architecture:
    1. Spatial Joint Embedding: Linear(2 -> d_joint) + Learnable Joint Positional Embedding.
    2. Spatial Self-Attention: Models anatomical correlations among 12 joints per frame.
    3. Temporal BiLSTM Backbone: Models sequential motion dynamics across time.
    4. Temporal Multi-Head Attention + Masked Pooling + Classifier Head.
    """

    def __init__(
        self,
        num_joints: int = 12,
        joint_in_dim: int = 2,  # (x, y) coordinates
        spatial_dim: int = 64,  # joint feature dimension
        spatial_heads: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 15,
        temporal_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.joint_in_dim = joint_in_dim
        self.spatial_dim = spatial_dim
        self.num_classes = num_classes

        # 1. Spatial Embedding: project 2D coordinate to spatial_dim
        self.joint_proj = nn.Linear(joint_in_dim, spatial_dim)
        self.joint_pos_embed = nn.Parameter(torch.randn(1, 1, num_joints, spatial_dim) * 0.02)

        # 2. Spatial Self-Attention across 12 body joints
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=spatial_dim,
            num_heads=spatial_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(spatial_dim)

        # Dimension entering the temporal LSTM: spatial_dim * num_joints or pooled
        # Flattening 12 joints * spatial_dim / 2 or pooling
        self.spatial_to_lstm = nn.Linear(spatial_dim * num_joints, spatial_dim * 2)

        # 3. Temporal BiLSTM Backbone
        directions = 2 if bidirectional else 1
        output_size = directions * hidden_size
        self.lstm = nn.LSTM(
            input_size=spatial_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Temporal Multi-Head Attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 5. Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(
        self,
        inputs: Tensor,
        lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        inputs: [Batch, Time, 24] or [Batch, Time, 12, 2]
        """
        if inputs.ndim == 3 and inputs.shape[-1] == 24:
            batch, time, _ = inputs.shape
            inputs = inputs.view(batch, time, self.num_joints, 2)
        else:
            batch, time, _, _ = inputs.shape

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

        # Step 1: Spatial Joint Embedding: [B, T, 12, 2] -> [B, T, 12, spatial_dim]
        spatial_feats = self.joint_proj(inputs) + self.joint_pos_embed

        # Reshape to apply spatial attention: [B * T, 12, spatial_dim]
        BT = batch * time
        spatial_in = spatial_feats.view(BT, self.num_joints, self.spatial_dim)
        spatial_out, _ = self.spatial_attention(spatial_in, spatial_in, spatial_in)
        spatial_out = self.spatial_norm(spatial_in + spatial_out)  # Residual + Norm

        # Flatten spatial joints: [B, T, num_joints * spatial_dim] -> project to [B, T, 128]
        temporal_inputs = spatial_out.view(batch, time, self.num_joints * self.spatial_dim)
        temporal_inputs = self.spatial_to_lstm(temporal_inputs)

        # Step 2: Temporal BiLSTM
        packed = pack_padded_sequence(
            temporal_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=time)

        # Step 3: Temporal Multi-Head Attention
        attended, _ = self.temporal_attention(
            lstm_output, lstm_output, lstm_output, key_padding_mask=padding_mask
        )

        # Masked Mean Pooling
        valid = (~padding_mask).unsqueeze(-1).to(attended.dtype)
        pooled = (attended * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        # Step 4: Classification
        logits = self.classifier(self.dropout(pooled))
        return logits
