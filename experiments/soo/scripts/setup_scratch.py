"""One-time (per-machine) setup: redirect experiments/soo/results into
~/nobackup/autodelete so run artifacts aren't written into the backed-up repo.

Usage: python experiments/soo/scripts/setup_scratch.py
Safe to re-run; no-ops if the symlink already exists.
"""

from selfconcept.common.paths import experiment_dir, scratch_dir

LINKED_DIRS = ["results", "slurm-logs"]


def main() -> None:
    for name in LINKED_DIRS:
        link = experiment_dir("soo") / name
        if link.is_symlink() or link.exists():
            print(f"{link} already exists, skipping")
            continue
        target = scratch_dir("soo") / name
        target.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        print(f"linked {link} -> {target}")


if __name__ == "__main__":
    main()
