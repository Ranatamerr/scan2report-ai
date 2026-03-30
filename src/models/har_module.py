#src/models/har_module.py
import torch
import torch.nn as nn

class HARModule(nn.Module):
    def __init__(self, hidden_size=768, num_queries=8, num_heads=8):
        super().__init__()

        self.hidden_size = hidden_size

        # 🔷 Learnable anatomical queries (organs)
        self.anatomy_queries = nn.Parameter(
            torch.randn(num_queries, hidden_size)
        )

        # 🔷 Cross-attention (queries attend to patches)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        # 🔷 Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, vit_tokens):
        """
        vit_tokens: [B, 197, 768]
        """

        # ❗ Remove CLS token
        patch_tokens = vit_tokens[:, 1:, :]  # [B, 196, 768]

        B = patch_tokens.size(0)

        # 🔷 Expand queries for batch
        queries = self.anatomy_queries.unsqueeze(0).expand(B, -1, -1)
        # shape: [B, num_queries, 768]

        # 🔷 Cross-attention: queries attend to patches
        attn_output, _ = self.cross_attn(
            query=queries,
            key=patch_tokens,
            value=patch_tokens
        )

        # 🔷 Residual + norm
        x = self.norm1(attn_output + queries)

        # 🔷 Feed-forward
        ffn_output = self.ffn(x)
        out = self.norm2(ffn_output + x)

        return out  # [B, num_queries, 768]