# models/decoder/layers.py

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1):
        super().__init__()

        # Self-attention (text side)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (CMM memory)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        """
        x: [B, T, D]  (text)
        memory: [B, S, D] (CMM output)
        """

        # 1. Self-attention (causal text modeling)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-attention (image memory)
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x