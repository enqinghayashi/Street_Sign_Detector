#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of current dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Ensure output dir exists and run prediction
python3 code/test/predict.py

echo "[INFO] Done. result.tsv is in prediction_result/"
