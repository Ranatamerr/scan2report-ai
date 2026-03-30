# src/data/dataset.py

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



import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io

class MimicDataset(Dataset):
    def __init__(self, parquet_paths, tokenizer, transform):
        # Load parquet files
        dfs = [pd.read_parquet(p) for p in parquet_paths]
        self.df = pd.concat(dfs, ignore_index=True)

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def decode_image(self, image_bytes):
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 🖼️ Decode image
        image = self.decode_image(row["image"])
        image = self.transform(image)

        # 📝 Use impression
        report = row["impression"]

        tokens = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }