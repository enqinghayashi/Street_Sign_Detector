from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(REPO_ROOT / "scripts" / "predict_yolo_reader.py"), run_name="__main__")
