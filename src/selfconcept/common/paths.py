"""Filesystem paths shared across experiments."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def experiment_dir(name: str) -> Path:
    return EXPERIMENTS_DIR / name
