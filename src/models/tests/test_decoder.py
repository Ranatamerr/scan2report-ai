import torch
from src.models.decoder.decoder import ReportDecoder

# fake settings
B = 2
T = 10
S = 8
vocab_size = 30522

# create model
model = ReportDecoder(vocab_size=vocab_size)

# dummy inputs
input_ids = torch.randint(0, vocab_size, (B, T))

# fake CMM memory (IMPORTANT)
memory = torch.randn(B, S, 768)

# forward pass
logits = model(input_ids, memory)

print("Input shape:", input_ids.shape)
print("Memory shape:", memory.shape)
print("Output shape:", logits.shape)