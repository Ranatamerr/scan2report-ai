import torch
from src.models.losses.contrastive_loss import contrastive_loss

vision = torch.randn(4, 768)
text = torch.randn(4, 768)

loss = contrastive_loss(vision, text)

print("Loss:", loss.item())