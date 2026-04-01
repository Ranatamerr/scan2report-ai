import torch
from src.models.vit_encoder import ViTEncoder
from src.models.har_module import HARModule

vit = ViTEncoder()
har = HARModule()

dummy = torch.randn(2, 3, 224, 224)

vit_out = vit(dummy)
har_out = har(vit_out)

print("ViT output:", vit_out.shape)
print("HAR output:", har_out.shape)