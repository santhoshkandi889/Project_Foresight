"""
Project FORESIGHT – main.py
Entry point for the full pipeline.

Usage:
  python main.py                  # full pipeline including training
  python main.py --skip-training  # skip model training (use saved models)
"""

import sys
from src.pipeline import run_pipeline

if __name__ == "__main__":
    skip_train = "--skip-training" in sys.argv
    run_pipeline(skip_training=skip_train)
