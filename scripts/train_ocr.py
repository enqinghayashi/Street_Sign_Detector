import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from pathlib import Path
import argparse


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from streetsign_detector.ocr.dataset import MCharDataset
from streetsign_detector.ocr.model import OCRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCRModel (EfficientNet backbone + 5 heads).")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tcdata", default=str(REPO_ROOT / "tcdata"), help="Path to tcdata directory")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "user_data" / "model_data"), help="Where to save checkpoints")
    return parser.parse_args()

def train():
    args = parse_args()

    # --- CONFIG ---
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    # NEW: Optimized Image Size (Height=64, Width=128)
    # House numbers are wide. 64x128 is a very standard OCR resolution.
    IMG_H, IMG_W = args.img_h, args.img_w

    # Paths
    tcdata_dir = args.tcdata
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    resume_path = os.path.join(output_dir, "latest.pth")
    best_model_path = os.path.join(output_dir, "model.dat")

    # --- TRANSFORMS ---
    # We use a slightly more aggressive augmentation
    train_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.1),
            shear=5  # Added shear
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- DATASET LOADING ---
    train_img = os.path.join(tcdata_dir, "mchar_train")
    if not os.path.exists(train_img): train_img += ".zip"

    val_img = os.path.join(tcdata_dir, "mchar_val")
    if not os.path.exists(val_img): val_img += ".zip"

    train_json = os.path.join(tcdata_dir, "mchar_train.json")
    val_json = os.path.join(tcdata_dir, "mchar_val.json")

    if not os.path.exists(train_json):
        print(f"Dataset JSON not found at {train_json}")
        return

    train_set = MCharDataset(train_img, train_json, transform=train_tf)
    val_set = MCharDataset(val_img, val_json, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=args.num_workers)

    # --- MODEL SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = OCRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Scheduler: Drop LR if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # NEW: Label Smoothing
    # This helps the model generalize better by not forcing 100% confidence
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)





if __name__ == "__main__":
    train()
