# src/data/test_dataset.py
# 🔥 ADD THIS PART HERE FIRST
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

img = Image.open("src/data/mimic/test.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor = transform(img)

print("Image tensor shape:", img_tensor.shape)
print("Min value:", img_tensor.min().item())
print("Max value:", img_tensor.max().item())

plt.imshow(img)
plt.title("Original Image")
plt.show()


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import os
from torchvision import transforms
from transformers import AutoTokenizer

from dataset import MimicDataset  # adjust import if needed


model_path = "./src/models/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)
# ✅ image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ dataset
dataset = MimicDataset(
    parquet_paths=[
        "src/data/mimic/train-00000-of-00002.parquet",
        "src/data/mimic/train-00001-of-00002.parquet"
    ],
    tokenizer=tokenizer,
    transform=transform
)

# ✅ get one sample
sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention mask shape:", sample["attention_mask"].shape)

print("First 10 tokens:", sample["input_ids"][:10])
