#!/bin/bash
set -e

# Legacy entrypoint wrapper (kept for backward compatibility)
# Runs the recommended YOLO + reader pipeline.

python code/test/predict.py
