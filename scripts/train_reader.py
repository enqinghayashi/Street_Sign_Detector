import os
import json
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 single-digit reader (digit crops from JSON boxes).")
    parser.add_argument("--train-zip", default=str(REPO_ROOT / "tcdata" / "mchar_train.zip"), help="Path to mchar_train.zip")
    parser.add_argument("--train-json", default=str(REPO_ROOT / "tcdata" / "mchar_train.json"), help="Path to mchar_train.json")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "user_data" / "model_data" / "single_digit_best.pth"),
        help="Where to save the trained weights.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()

# Dataset extraction into single digit samples
class SingleDigitCropDataset(Dataset):
    def __init__(self, json_path, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.samples = [] 
        
        # Load JSON and flatten it into individual digit samples
        # Format: (filename, box_params, label_value)
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        print("Parsing JSON to create crop list...")
        for filename, data in meta.items():
            # data contains: 'height', 'label', 'left', 'top', 'width' lists
            labels = data['label']
            tops = data['top']
            lefts = data['left']
            heights = data['height']
            widths = data['width']
            
            for i in range(len(labels)):
                # Create a sample for every single digit
                label = int(labels[i])
                
                # Fix label 10 -> 0 if necessary 
                # Tianchi usually uses 0-9 directly, but just in-case
                if label == 10: label = 0
                
                box = (lefts[i], tops[i], widths[i], heights[i])
                self.samples.append((filename, box, label))
                
        print(f"Found {len(self.samples)} individual digits.")
        self.zip_handle = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.zip_handle is None:
            self.zip_handle = zipfile.ZipFile(self.zip_path, 'r')
            
        filename, (x, y, w, h), label = self.samples[idx]
        
        # Read Image from Zip
        # Check if file is in folder (e.g. "mchar_train/000000.png")
        try:
            data = self.zip_handle.read(filename)
        except KeyError:
            # Fallback for folder inside zip
            data = self.zip_handle.read(f"mchar_train/{filename}")
            
        img = Image.open(io.BytesIO(data)).convert("RGB")
        
        # Crop the digit
        # Add a tiny bit of padding (optional, helps context)
        img_w, img_h = img.size
        crop = img.crop((max(0, x), max(0, y), min(img_w, x+w), min(img_h, y+h)))
        
        if self.transform:
            crop = self.transform(crop)
            
        return crop, label

# Standard ImageNet normalization
train_tf = transforms.Compose([
    transforms.Resize((128, 128)), # Reshape to square so it's easier for EfficiencyNet to detect
    transforms.RandomRotation(10), # Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model
def get_reader_model():
    model = models.efficientnet_b0(weights='DEFAULT')
    # Change classifier for 10 classes (0-9)
    # EfficientNet B0 classifier layer structure:
    # (1): Linear(in_features=1280, out_features=1000, bias=True)
    model.classifier[1] = nn.Linear(1280, 10) 
    return model

# Trainin logic
def train_reader():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_zip = args.train_zip
    train_json = args.train_json
    output_path = args.output

    if not os.path.exists(train_zip):
        raise FileNotFoundError(f"Train zip not found: {train_zip}")
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Train JSON not found: {train_json}")


if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    train_reader()