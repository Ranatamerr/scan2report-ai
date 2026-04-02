import torch
from src.models.model import MedicalReportModel

# settings
B = 2
T = 12
vocab_size = 30522

# model
model = MedicalReportModel(vocab_size=vocab_size)

# dummy inputs
images = torch.randn(B, 3, 224, 224)
input_ids = torch.randint(0, vocab_size, (B, T))

# forward pass
logits = model(images, input_ids)

print("Images:", images.shape)
print("Input IDs:", input_ids.shape)
print("Output:", logits.shape)