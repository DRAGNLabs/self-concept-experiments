"""Filesystem paths shared across experiments."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCRATCH_DIR = Path.home() / "nobackup" / "autodelete" / REPO_ROOT.name


def experiment_dir(name: str) -> Path:
    return EXPERIMENTS_DIR / name


def scratch_dir(name: str) -> Path:
    """Autodelete scratch space for one experiment's heavy/transient artifacts."""
    return SCRATCH_DIR / name
