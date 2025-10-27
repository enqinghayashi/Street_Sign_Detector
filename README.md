# Street_Sign_Detector

- tcdata/: empty on submission; test data will be placed here.
- user_data/: stores trained models and temporary/intermediate files.
- prediction_result/: stores the final result file result.tsv.
- code/: contains training and prediction code, and entry scripts (train.sh, test.sh).

Environment
- OS: Linux or Windows
- Python: 3.8+
- Packages: standard library only (no extra packages required)

Folder Structure
- README.md
- requirements.txt
- tcdata/
- user_data/
  - model_data/
  - tmp_data/
- prediction_result/
- code/
  - train/
    - train.py
  - test/
    - predict.py
  - train.sh
  - test.sh

How to Train
- Fixed hyperparameters are embedded in code/train/train.py.
- From repository root:
  - bash code/train.sh
- Output:
  - user_data/model_data/model.json (toy baseline model metadata)

How to Predict (Required)
- This is the official entrypoint.
- From repository root:
  - bash code/test.sh
- Output:
  - prediction_result/result.tsv
- The script:
  - Recursively searches for zip files under tcdata/ (e.g., tcdata/街景识别/test_a.zip, test_b.zip).
  - Extracts them into user_data/tmp_data/extracted/.
  - Scans images and writes a TSV with columns:
    - image_name (relative path within the archive)
    - prediction (baseline placeholder)
- Note: The included baseline emits a constant placeholder label "unknown". Replace logic in code/test/predict.py and code/train/train.py with your actual model.

Paths
- All file paths are relative to repository root in code.
- No absolute paths are used in code execution.

Reproducibility
- No randomness used in the baseline.
- No external data or internet access required.

Training Model
- Trained weights is under user_data/model_data/ (e.g., model.pth).
- Implement real training in code/train/train.py.
- Implement real inference in code/test/predict.py, ensuring you still:
  - Read test data from ../tcdata
  - Write final TSV to ../prediction_result/result.tsv
  - Keep the result file format identical (image_name<TAB>prediction)

