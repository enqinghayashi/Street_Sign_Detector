# Street_Sign_Detector (SVHN Tianchi 531795)

- Competition: https://tianchi.aliyun.com/competition/entrance/531795
- Task: Recognize all digits in each image (sequence-level accuracy).
- Provided: images + bounding boxes + labels (train/val), boxes (test A).

## Folder layout
project/
- data/
  - raw/
    - train/            # unzip mchar_train.zip here
    - val/              # unzip mchar_val.zip here
    - test_a/           # unzip mchar_test_a.zip here
    - annotations/      # put mchar_train.json + mchar_val.json here
    - sample_submit/    # put mchar_sample_submit_A.csv here
  - processed/          # optional: cached crops, tensors
  - interim/            # optional: intermediate artifacts
- src/                  # code
- notebooks/            # exploratory notebooks
- configs/              # training/inference configs
- models/               # trained weights (ignored by git)
- outputs/              # submissions, logs, figures (ignored by git)

## What to download (download all)
- mchar_train.zip
- mchar_train.json
- mchar_val.zip
- mchar_val.json
- mchar_test_a.zip
- mchar_sample_submit_A.csv

## Where to place files
- Unzip:
  - mchar_train.zip -> data/raw/train/
  - mchar_val.zip -> data/raw/val/
  - mchar_test_a.zip -> data/raw/test_a/
- Copy:
  - mchar_train.json -> data/raw/annotations/
  - mchar_val.json -> data/raw/annotations/
  - mchar_sample_submit_A.csv -> data/raw/sample_submit/

## Quick start
1) Create a Python 3.9–3.11 env and install: torch+torchvision, numpy, pandas, opencv-python, matplotlib, albumentations, tqdm. (Optional: pytorch-lightning, omegaconf/hydra, wandb/tensorboard.)
2) Sanity check data: open a few images, parse JSON, verify boxes and labels (0–9), ensure ~30k train, ~10k val, ~40k test A.
3) Baseline plan:
   - Crop characters by boxes (train/val).
   - Train a 10-class digit classifier.
   - For each image: sort boxes by left, predict digits, join as string.
4) Validation: exact string-match accuracy on val.
5) Inference: generate submission CSV (file_name,file_code) from test A and save to outputs/.

Notes:
- data, outputs, and models are ignored by Git. Keep only small configs/notebooks/code.
- See data/mchar_data_list_0515.csv for file links and target locations.



