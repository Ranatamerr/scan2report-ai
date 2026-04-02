import torch
import torch.nn as nn
from transformers import AutoModel

from .vit_encoder import ViTEncoder
from .har_module import HARModule
from .cmm import CMMModule
from .decoder.decoder import ReportDecoder


class MedicalReportModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()

        # -------------------------
        # 1. Vision Encoder (ViT)
        # -------------------------
        self.vision_encoder = ViTEncoder()

        # -------------------------
        # 2. HAR Module
        # -------------------------
        self.har = HARModule()

        # -------------------------
        # 3. Text Encoder (BERT)
        # -------------------------
        #self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_encoder = nn.Embedding(vocab_size, hidden_dim)

        # -------------------------
        # 4. CMM Module
        # -------------------------
        self.cmm = CMMModule()

        # -------------------------
        # 5. Decoder
        # -------------------------
        self.decoder = ReportDecoder(
            vocab_size=vocab_size,
            dim=hidden_dim
        )

    def forward(self, images, input_ids, attention_mask=None):

        # -------------------------
        # 1. Vision Encoding
        # -------------------------
        vit_out = self.vision_encoder(images)
        # [B, 197, 768]

        # -------------------------
        # 2. HAR
        # -------------------------
        har_out = self.har(vit_out)
        # [B, 8, 768]

        # -------------------------
        # 3. Text Encoding (FIXED)
        # -------------------------
        #text_outputs = self.text_encoder(input_ids=input_ids)
        #text_tokens = text_outputs.last_hidden_state
        text_tokens = self.text_encoder(input_ids)
        # [B, T, 768]

        # -------------------------
        # 4. CMM
        # -------------------------
        cmm_out = self.cmm(har_out, text_tokens)
        # [B, 8, 768]

        # -------------------------
        # 5. Decoder
        # -------------------------
        logits = self.decoder(input_ids, cmm_out)
        # [B, T, vocab_size]

        return logits