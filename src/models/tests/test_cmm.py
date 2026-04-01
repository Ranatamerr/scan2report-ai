import torch

from src.models.vit_encoder import ViTEncoder
from src.models.har_module import HARModule
from src.models.cmm import CMM


# --------------------
# 1. Models
# --------------------
vit = ViTEncoder()
har = HARModule()
cmm = CMM()


# --------------------
# 2. Dummy image
# --------------------
images = torch.randn(2, 3, 224, 224)

# --------------------
# 3. Dummy text (simulating report embeddings)
# --------------------
text_tokens = torch.randn(2, 20, 768)


# --------------------
# 4. Forward pass
# --------------------
vit_out = vit(images)         # [2, 197, 768]
har_out = har(vit_out)        # [2, 8, 768]
cmm_out = cmm(har_out, text_tokens)


# --------------------
# 5. Print shapes
# --------------------
print("ViT:", vit_out.shape)
print("HAR:", har_out.shape)
print("CMM:", cmm_out.shape)