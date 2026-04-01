# models/decoder/decoder.py

import torch
import torch.nn as nn
from .layers import DecoderLayer
from .pos_encoding import PositionalEncoding


class ReportDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=768,
        num_layers=4,
        max_len=128,
        heads=8
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim) #embedding layer to convert token IDs to vectors
        self.pos_enc = PositionalEncoding(dim, max_len) #positional encoding to add position information to the input

        self.layers = nn.ModuleList([ #list of decoder layers to process the input
            DecoderLayer(dim=dim, heads=heads)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(dim, vocab_size)

        self.max_len = max_len

    def generate_causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, device=device) * float("-inf"),
            diagonal=1
        )

    def forward(self, input_ids, memory):
        """
        input_ids: [B, T]
        memory: [B, S, 768] (from CMM)
        """

        B, T = input_ids.shape

        x = self.token_emb(input_ids) #convert token IDs to vectors
        x = self.pos_enc(x) #add position information to the input

        # causal mask (prevents cheating)
        mask = self.generate_causal_mask(T, x.device) #generate a causal mask to prevent the model from cheating

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask) #process the input through the decoder layers

        logits = self.lm_head(x) #convert the output to logits

        return logits