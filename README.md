# Street Sign Detector (YOLO + EfficientNet Fusion)

This project detects and reads multi-digit street/house-number signs using a two-stage pipeline:

1) **YOLO (Ultralytics)** detects digit bounding boxes.
2) **EfficientNet-B0** classifies each cropped digit, then digits are sorted left-to-right and concatenated.

An alternative baseline also exists: a single EfficientNet backbone with **5 classification heads** (predicts up to 5 digits, padded), trained end-to-end as `OCRModel`.

## Repository Layout

- src/streetsign_detector/: importable Python package (models/datasets)
- scripts/: runnable entrypoints for training and inference
- notebooks/: exploration + training notebooks (moved from the old code/ folder)
- tcdata/: dataset inputs (zips/json) (typically **not committed**)
- user_data/model_data/: trained weights/checkpoints
- runs/: Ultralytics training outputs (YOLO weights typically live under runs/detect/...)
- prediction_result/: CSV outputs
- weights/: misc pretrained weights (e.g., yolov8n.pt)

Legacy entrypoints are kept under `code/` (e.g. `bash code/test.sh`).

The old `code/` folder is kept as **compatibility wrappers** so older commands still run.

## Setup

- Python: 3.10+ recommended
- Install deps:
  - `pip install -r requirements.txt`

If you run scripts directly (without installing as a package), they already bootstrap `src/` onto `sys.path`.

## Training

### A) Train the single-digit EfficientNet reader

Trains EfficientNet-B0 on digit crops derived from `tcdata/mchar_train.json` + `tcdata/mchar_train.zip`.

- `python scripts/train_reader.py`
- Output: `user_data/model_data/single_digit_best.pth`

### B) Train the multi-head OCRModel (optional)

Trains an EfficientNet-based model that predicts a 5-digit sequence directly.

- `python scripts/train_ocr.py`
- Output: `user_data/model_data/model.dat` (best) and `user_data/model_data/latest.pth` (resume)

### C) Train YOLO

YOLO training is currently driven from the notebooks under `notebooks/training/` (Ultralytics).
After training, set `--yolo-weights` to the produced `best.pt`.

## Inference

### Recommended: YOLO detector + EfficientNet reader

- `python scripts/predict_yolo_reader.py \
    --test-zip tcdata/mchar_test_a.zip \
    --yolo-weights runs/detect/number_v8/weights/best.pt \
    --reader-weights user_data/model_data/single_digit_best.pth \
    --output-csv prediction_result/result.csv`

The script writes a CSV with header `file_name,file_code`.

### Alternative: OCRModel (no YOLO)

- `python scripts/predict_ocr.py`
- Expects: `user_data/model_data/model.dat`
- Writes: `prediction_result/result.csv`

## Notes for GitHub

- Don’t push large generated folders (virtualenvs, datasets, `runs/`, outputs). Use `.gitignore` or Git LFS for large weights.
- If you want, I can add a clean `.gitignore` + optional Git LFS setup before we push.


