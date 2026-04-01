import torch
from src.models.vit_encoder import ViTEncoder

def main():
    vit = ViTEncoder()

    dummy = torch.randn(2, 3, 224, 224)

    out = vit(dummy)

    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()