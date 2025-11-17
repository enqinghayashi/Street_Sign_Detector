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











if __name__ == "__main__":
    train()
