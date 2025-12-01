import torch
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from streetsign_detector.constants import PADDING_TOKEN
from streetsign_detector.ocr.model import OCRModel
from streetsign_detector.ocr.zip_test_dataset import ZipTestDataset


def _default_test_zip(repo_root: Path) -> str:
    for candidate in [repo_root / "tcdata" / "mchar_test_a.zip", repo_root / "tcdata" / "test_a.zip"]:
        if candidate.exists():
            return str(candidate)
    return str(repo_root / "tcdata" / "mchar_test_a.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference using OCRModel (multi-head EfficientNet baseline).")
    parser.add_argument("--test-zip", default=_default_test_zip(REPO_ROOT), help="Path to test zip.")
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "user_data" / "model_data" / "model.dat"),
        help="Path to OCRModel weights/checkpoint.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "prediction_result" / "result.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()

def predict():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    model_path = args.model_path
    zip_file = args.test_zip

    if not os.path.exists(zip_file):
        raise FileNotFoundError(f"Test zip not found: {zip_file}")

    # Load Model
    model = OCRModel().to(device)
    if not os.path.exists(model_path):
        print("Error: Model file not found. Train first.")
        return

    checkpoint = torch.load(model_path, map_location=device)

    # Flexible loading (handle if saved as full dict or just weights)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Dataset
    test_set = ZipTestDataset(zip_file)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = []
    print(f"Inference on {len(test_set)} images...")

    with torch.no_grad():
        for i, (img, names) in enumerate(loader):
            img = img.to(device)

            # Forward
            out = model(img) # List of 5 tensors
            out_stack = torch.stack(out, dim=1) # [Batch, 5, 11]

            # Greedy Decode
            preds = out_stack.argmax(dim=2).cpu().numpy()


if __name__ == "__main__":
    predict()
