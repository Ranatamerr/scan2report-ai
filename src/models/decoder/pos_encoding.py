# models/decoder/pos_encoding.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, dim) #create an array of zeros , empty table

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)) #creates frequency scaling for sine/cosine waves ,
        #low dims= slow variation, high dims= fast variation , giving each position a unique frequency 

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0)) #register buffer to make it part of the model's state

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, :x.size(1)] #add the positional encoding to the input