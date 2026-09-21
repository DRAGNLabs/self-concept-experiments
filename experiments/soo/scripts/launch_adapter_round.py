"""Freeze a snapshot for the existing-adapter round and submit it (ADAPTER_ROUND.md).

Usage, from the repository root:
  .venv/bin/python experiments/soo/scripts/launch_adapter_round.py            # freeze, check, submit
  .venv/bin/python experiments/soo/scripts/launch_adapter_round.py --no-submit
  .venv/bin/python experiments/soo/scripts/launch_adapter_round.py --submit-existing SNAPSHOT
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOO = ROOT / "experiments/soo"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYER = 32
ADAPTERS = {"original": "qwen38-27b-L32", "agentic": "qwen38-27b-agentic-L32", "mixed": "qwen38-27b-mixed-L32"}
TRAIN_FILES = {"original": "data/train_soo_pairs.jsonl", "agentic": "data/train_soo_pairs_agentic.jsonl", "mixed": "data/train_soo_pairs_mixed.jsonl"}
SEEDS = (0, 1, 2)
COMMON = (f"--model {MODEL} --revision {REVISION} --probes data/subspace_pilot_pairs.jsonl --split development "
          f"--layer {LAYER} --residual-layers 32 39 47 55 63 --positions all --token-mode last "
          "--batch-size 4 --device cuda --dtype bfloat16 --bootstrap 2000 --seed 0")

RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-adapters-qwen38
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root, passing an absolute frozen snapshot path.
# Keep the GPU assigned by Slurm; never borrow another job's GPU.
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit through Slurm from the repository root}
export PYTHONPATH="$SNAPSHOT_ROOT/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOO_CHAT_KWARGS='{"enable_thinking": false}'
export OMP_NUM_THREADS=4
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not assign a visible GPU; refusing to select an unallocated GPU."
    exit 1
fi
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd "$SNAPSHOT_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
mkdir -p output
run() {  # run NAME ARGS...
    local name=$1; shift
    if [[ -f "output/$name/manifest.json" ]]; then echo "skip $name (complete)"; return; fi
    rm -rf "output/$name"
    echo "=== $(date -u +%FT%TZ) start $name"
    "$PY" -u -m selfconcept.soo.measure_overlap __COMMON__ "$@" --out "output/$name"
    echo "=== $(date -u +%FT%TZ) done $name"
}
run steering-L32-add-a10 --vectors data/qwen38_27b.pt --mode add --alpha 10 --random-seeds 0 1 2
__ADAPTER_RUNS__
"$PY" analyze_adapter_round.py output | tee output/analysis.log
echo "=== all runs complete"
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return " ".join(text.split())


def check_membership():
    """No development prompt may occur in any adapter training file."""
    rows = [json.loads(l) for l in (SOO / "data/subspace_pilot_pairs.jsonl").read_text().splitlines() if l.strip()]
    dev = [r for r in rows if r["split"] == "development"]
    dev_prompts = {norm(r[k]) for r in dev for k in ("self_prompt", "other_prompt")}
    report = {"development_pairs": len(dev), "development_prompts": len(dev_prompts), "hits": {}}
    for label, rel in TRAIN_FILES.items():
        train = [json.loads(l) for l in (SOO / rel).read_text().splitlines() if l.strip()]
        train_prompts = {norm(r[k]) for r in train for k in ("self_prompt", "other_prompt")}
        hits = sorted(dev_prompts & train_prompts)
        report["hits"][label] = {"file": rel, "train_pairs": len(train), "sha256": sha256(SOO / rel), "literal_hits": hits}
        if hits:
            raise SystemExit(f"development prompt found in {rel}: {hits[:3]}")
    report["thematic_overlap_note"] = "development family 'telescope' vs training item 'professional telescope'; prompts and task differ"
    return report


def check_adapters():
    info = {}
    for label, name in ADAPTERS.items():
        for seed in SEEDS:
            d = SOO / "results/checkpoints" / name / f"seed{seed}"
            log = json.loads((d / "train_log.json").read_text())
            cfg = log["config"]
            if cfg["layer"] != LAYER or cfg["train_data"] != TRAIN_FILES[label] or log["seed"] != seed or cfg["model"] != MODEL:
                raise SystemExit(f"unexpected training record for {d}: {cfg}, seed {log['seed']}")
            info[f"{label}/seed{seed}"] = {"path": str(d.relative_to(ROOT)), "config": cfg, "seed": log["seed"],
                                          "final_epoch_loss": log["epoch_losses"][-1], "n_epoch_losses": len(log["epoch_losses"])}
    return info


def freeze(snapshot):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    shutil.copy2(SOO / "data/subspace_pilot_pairs.jsonl", snapshot / "data/subspace_pilot_pairs.jsonl")
    shutil.copy2(SOO / "results/steering/qwen38_27b.pt", snapshot / "data/qwen38_27b.pt")
    for label, name in ADAPTERS.items():
        for seed in SEEDS:
            shutil.copytree(SOO / "results/checkpoints" / name / f"seed{seed}", snapshot / "adapters" / label / f"seed{seed}")
    shutil.copy2(SOO / "ADAPTER_ROUND.md", snapshot / "ADAPTER_ROUND.md")
    shutil.copy2(SOO / "scripts/analyze_adapter_round.py", snapshot / "analyze_adapter_round.py")
    shutil.copy2(Path(__file__), snapshot / "launch_adapter_round.py")
    adapter_runs = "\n".join(f"run adapter-{label}-seed{seed} --adapter adapters/{label}/seed{seed}"
                             for label in ADAPTERS for seed in SEEDS)
    (snapshot / "run.sh").write_text(RUN_SH.replace("__COMMON__", COMMON).replace("__ADAPTER_RUNS__", adapter_runs))
    (snapshot / "run.sh").chmod(0o755)


def software_checks(snapshot):
    env = {"PYTHONPATH": str(snapshot / "src"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", str(snapshot / "tests"), "-p", "test_soo*.py", "-v"],
                          cwd=snapshot, env=env, text=True, capture_output=True)
    (snapshot / "software_checks.log").write_text(proc.stdout + proc.stderr)
    m = re.search(r"Ran (\d+) tests", proc.stderr + proc.stdout)
    if proc.returncode != 0 or not m:
        raise SystemExit(f"software checks failed; see {snapshot / 'software_checks.log'}")
    return f"{m.group(1)} tests passed on CPU; see software_checks.log"


def submit(snapshot):
    proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot)], cwd=ROOT, text=True, capture_output=True, check=True)
    job = proc.stdout.strip().split(";")[0]
    (snapshot / "submission.json").write_text(json.dumps({
        "job_id": job, "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(SOO / "slurm-logs" / f"soo2-adapters-qwen38-{job}.out")}, indent=2) + "\n")
    print(f"submitted job {job}")
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--submit-existing", type=Path)
    args = ap.parse_args()
    if args.submit_existing:
        snap = args.submit_existing.resolve()
        if not (snap / "launch.json").exists() or (snap / "submission.json").exists():
            raise SystemExit("snapshot must be frozen and not yet submitted")
        submit(snap)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = SOO / "results/study2" / f"adapters-qwen38-L{LAYER}-{stamp}"
    membership = check_membership()
    adapters = check_adapters()
    freeze(snapshot)
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty,
        "plan": "ADAPTER_ROUND.md", "model": MODEL, "model_revision": REVISION, "layer": LAYER,
        "expected_runs": ["steering-L32-add-a10"] + [f"adapter-{l}-seed{s}" for l in ADAPTERS for s in SEEDS],
        "development_pairs": 64, "training_membership_check": membership, "adapters": adapters,
        "test_result": test_result,
        "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        submit(snapshot)


if __name__ == "__main__":
    main()
