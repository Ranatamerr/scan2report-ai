import torch
import torch.nn as nn
from transformers import ViTModel

class ViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(model_name)
        
        # output dimension = 768 for vit-base
        self.hidden_size = self.vit.config.hidden_size

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        """

        outputs = self.vit(pixel_values=images)

        # patch embeddings (NOT classification head)
        # shape: [B, num_patches+1, hidden]
        sequence_output = outputs.last_hidden_state

        return sequence_output