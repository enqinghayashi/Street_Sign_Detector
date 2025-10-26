# Tianchi SVHN Group Project (2025)

Street View House Numbers (SVHN) competition on Tianchi.

- Link: https://tianchi.aliyun.com/competition/entrance/531795
- Data: Train 30k, Val 10k, TestA 40k, TestB 40k. Boxes + labels for train/val; boxes for test.
- Task: Recognize all digits in each image (sequence).
- Metric: Exact sequence accuracy (predicted string equals ground truth string).
- GPU: 60 hours on Tianchi Notebook.

Repo layout
- data/           Local dataset (ignored by git). Keep structure you download.
- checkpoints/    Trained weights (ignored).
- outputs/        Predictions, submissions, logs (ignored).
- README.md       This file.
- requirements.txt Minimal dependencies to start.
- LICENSE         MIT.

Quickstart
1) Python 3.10+ recommended.
2) Create env and install deps:
   - python -m venv .venv
   - .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)
   - pip install -r requirements.txt
3) Put datasets under data/ (do not commit). Example:
   - data/images/... (train/val/test images)
   - data/annotations.csv (if using a unified CSV; columns: image_path,x1,y1,w,h,label,set)
4) Code will be added incrementally (baseline: crop digits by boxes, train digit classifier, decode by x-order). Use outputs/ for submissions.

Team workflow (3 people)
- Branching: main (protected), feature branches per task (feat/data, feat/model, feat/infra).
- Reviews: open PRs into main, require 1 reviewer.
- Issues: track tasks with labels (data, model, infra, eval, submit).
- Style: black (line length 100) + isort; type hints where practical.

Submission format
- CSV with columns: image_name,label
- label is the full predicted string per image (e.g., 3075)
