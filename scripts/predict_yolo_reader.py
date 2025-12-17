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
    parser.add_argument("--iou", type=float, default=0.35, help="YOLO IoU threshold.")
    parser.add_argument("--conf", type=float, default=None, help="Optional YOLO confidence threshold.")
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
    
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return ""
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    detections = []
    img_w, img_h = pil_img.size
    
    # 2. Crop and Read
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Clip coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop
        crop = pil_img.crop((x1, y1, x2, y2))
        
        # Preprocess for EfficientNet
        input_tensor = reader_tf(crop).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = reader_model(input_tensor)
            _, pred = torch.max(output, 1)
            digit = pred.item()
            
        detections.append({
            "digit": digit,
            "x_center": (x1 + x2) / 2
        })
    
    if not detections:
        return ""
        
    # 3. Sort Left-to-Right
    detections.sort(key=lambda x: x["x_center"])
    
    # Concatenate
    final_str = "".join(str(d["digit"]) for d in detections)
    return final_str

# Main
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    if not os.path.exists(args.test_zip):
        raise FileNotFoundError(f"Test zip not found: {args.test_zip}")
    if not os.path.exists(args.yolo_weights):
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not os.path.exists(args.reader_weights):
        raise FileNotFoundError(f"Reader weights not found: {args.reader_weights}")

    print("Loading models...")
    yolo_model = YOLO(args.yolo_weights)

    reader_model = get_reader_model().to(device)
    reader_model.load_state_dict(torch.load(args.reader_weights, map_location=device))
    reader_model.eval()

    results = []
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    print(f"Reading from {args.test_zip}...")
    
    # Open Zip and iterate
    with zipfile.ZipFile(args.test_zip, 'r') as zf:
        # Filter for valid images only
        file_list = [f for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX')]
        file_list = sorted(file_list) # Sort filenames just in case
        
        total = len(file_list)
        print(f"Found {total} images. Starting Inference...")
        
        for i, filename in enumerate(file_list):
            # Read bytes
            img_data = zf.read(filename)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Predict
            pred_str = predict_image(
                pil_img=img,
                yolo_model=yolo_model,
                reader_model=reader_model,
                device=device,
                iou=float(args.iou),
                conf=args.conf,
            )
            
            # Clean filename (remove folder prefix if exists)
            clean_name = os.path.basename(filename)
            results.append(f"{clean_name},{pred_str}")
            
            if i % 100 == 0:
                print(f"Processing {i}/{total} | Last: {clean_name} -> {pred_str}")

    # Save result.csv
    with open(args.output_csv, "w") as f:
        f.write("file_name,file_code\n")
        f.write("\n".join(results))
        
    print(f"\nDONE! Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
