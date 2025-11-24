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

    # --- RESUME LOGIC ---
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        print("Starting training from scratch.")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            # Forward
            out = model(img) # list of 5 tensors
            out_stack = torch.stack(out, dim=1) # [Batch, 5, 11]

            # Compute loss
            # Permute to [Batch, Classes, Seq_Len] for CrossEntropy
            loss = loss_fn(out_stack.permute(0, 2, 1), label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"\rEpoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}", end="")

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} Avg Train Loss: {avg_loss:.4f} | LR: {current_lr:.1e}")

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)

                out = model(img)
                out_stack = torch.stack(out, dim=1)

                loss = loss_fn(out_stack.permute(0, 2, 1), label)
                val_loss += loss.item()

                # Calculate simple accuracy (perfect match of all digits)
                # preds: [Batch, 5]
                preds = out_stack.argmax(dim=2)

                # Check row by row equality
                # eq: [Batch, 5] -> all(dim=1) -> [Batch] boolean
                row_match = torch.all(preds.eq(label), dim=1)
                correct += row_match.sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Validation Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")


if __name__ == "__main__":
    train()
