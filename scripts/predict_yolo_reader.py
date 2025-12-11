import argparse
import io
import os
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_test_zip() -> str:
    for candidate in [
        REPO_ROOT / "tcdata" / "mchar_test_a.zip",
        REPO_ROOT / "tcdata" / "test_a.zip",
    ]:
        if candidate.exists():
            return str(candidate)
    return str(REPO_ROOT / "tcdata" / "mchar_test_a.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detection + EfficientNet single-digit reader.")
    parser.add_argument("--test-zip", default=_default_test_zip(), help="Path to test zip (mchar_test_a.zip / test_a.zip).")
    parser.add_argument(
        "--yolo-weights",
        default=str(REPO_ROOT / "runs" / "detect" / "number_v8" / "weights" / "best.pt"),
        help="Path to YOLO weights (.pt).",
    )
    parser.add_argument(
        "--reader-weights",
        default=str(REPO_ROOT / "user_data" / "model_data" / "single_digit_best.pth"),
        help="Path to EfficientNet reader weights (.pth).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "prediction_result" / "result.csv"),
        help="Where to write the CSV results.",
    )
    return parser.parse_args()

def get_reader_model():
    model = models.efficientnet_b0(weights=None) # We load our own weights here
    # Replace the final layer to match our 10-class training
    model.classifier[1] = nn.Linear(1280, 10) 
    return model

# Transforms 
reader_tf = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Pipeline Logics: YOLO then EfficiencyNet Classification
def predict_image(*, pil_img: Image.Image, yolo_model: YOLO, reader_model: nn.Module, device: torch.device, iou: float, conf: float | None):
    # 1. YOLO Detection
    kwargs = {"verbose": False, "iou": iou}
    if conf is not None:
        kwargs["conf"] = conf
    results = yolo_model(pil_img, **kwargs)
    
    
        
    
    

# Main
    


if __name__ == "__main__":
    main()
