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





if __name__ == "__main__":
    predict()
