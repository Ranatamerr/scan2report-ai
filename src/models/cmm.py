import torch
import torch.nn as nn

class CMM(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super().__init__()

        # 🔷 Cross-attention: vision queries text OR text queries vision
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)

        # 🔷 optional refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, vision_tokens, text_tokens=None):
        """
        vision_tokens: [B, 8, 768]  (from HAR)
        text_tokens:   [B, T, 768]   (report embeddings during training)
        """

        if text_tokens is None:
            # inference fallback: self-attend vision only
            attn_out, _ = self.cross_attn(
                vision_tokens, vision_tokens, vision_tokens
            )
        else:
            # 🔷 main case: vision attends to text
            attn_out, _ = self.cross_attn(
                query=vision_tokens,
                key=text_tokens,
                value=text_tokens
            )

        x = self.norm(vision_tokens + attn_out)

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)

        return out