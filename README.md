# Tianchi SVHN Group Project (2025)

Street View House Numbers (SVHN) competition on Tianchi.

- Link: https://tianchi.aliyun.com/competition/entrance/531795
- Data: Train 30k, Val 10k, TestA 40k, TestB 40k. Boxes + labels for train/val; boxes for test.
- Task: Recognize all digits in each image (sequence).
- Metric: Exact sequence accuracy (predicted string equals ground truth string).
- GPU: 60 hours on Tianchi Notebook.

Team workflow 
- Branching: main (protected), feature branches per task (feat/data, feat/model, feat/infra).
- Reviews: open PRs into main, require 1 reviewer.
- Issues: track tasks with labels (data, model, infra, eval, submit).
- Style: black (line length 100) + isort; type hints where practical.


