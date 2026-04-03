#!/bin/bash
set -e

# Legacy entrypoint wrapper (kept for backward compatibility)
# Runs the OCR training wrapper.

python code/train/train.py
