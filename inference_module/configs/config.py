"""
Configuration file for dynamic path management.
All paths should be absolute and resolved using pathlib.
"""

from pathlib import Path

# Project root directory (assuming this file is in configs/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Weights directory
WEIGHTS_DIR = PROJECT_ROOT / "weights"

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Weights Directory: {WEIGHTS_DIR}")
