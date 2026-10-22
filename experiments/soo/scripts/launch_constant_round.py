"""Freeze and submit the constant-replacement round for one model (CONSTANT_ROUND.md).

Usage, from the repository root:
  .venv/bin/python experiments/soo/scripts/launch_constant_round.py --model-key gemma4-31b [--no-submit]
The job measures the constants (make_constants.py), then evaluates twelve conditions in both
orientations on main / treasure_hunt / perspectives (n=250), then runs the pre-specified analysis.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOO = ROOT / "experiments/soo"
TRAIN_FILE = "data/train_soo_pairs.jsonl"

MODELS = {
    "gemma4-31b": dict(model="google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", layer=32,
                       adapter_dir="results/checkpoints/gemma4-31b-L32", seeds=(0, 1, 2), suffix="room_only", hours=6,
                       first_study="main deceptive 100/100 base, 0/0 adapter seed0 (orig/mirrored, n=250)"),
    "gemma4-12b": dict(model="google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", layer=24,
                       adapter_dir="results/checkpoints/gemma4-12b-L24", seeds=(0, 1, 2), suffix="room_only", hours=6,
                       first_study="main deceptive 100/100 base, 5/7 adapter seed0 (orig/mirrored, n=250)"),
    "mistral-7b": dict(model="mistralai/Mistral-7B-Instruct-v0.2", revision="63a8b081895390a26e140280378bc85ec8bce07a", layer=16,
                       adapter_dir="results/checkpoints/mistral-7b-lasttok-L16", seeds=(0, 1, 2), suffix="i_would", hours=10,
                       first_study="main deceptive 90.4 base, 9.9 ± 6.1 over five seeds (n=250)"),
}

CONDITIONS = [
    ("base", ""),
    ("adapter-seed0", "--adapter adapters/seed0"),
    ("replace-adapter-s0", "__R__ --steer-constant adapter_seed0 --steer-positions from_last"),
    ("replace-adapter-s1", "__R__ --steer-constant adapter_seed1 --steer-positions from_last"),
    ("replace-adapter-s2", "__R__ --steer-constant adapter_seed2 --steer-positions from_last"),
    ("replace-base-mean", "__R__ --steer-constant base_mean --steer-positions from_last"),
    ("replace-zero", "__R__ --steer-constant zero --steer-positions from_last"),
    ("replace-random-s0", "__R__ --steer-constant adapter_seed0 --steer-random-seed 0 --steer-positions from_last"),
    ("replace-random-s1", "__R__ --steer-constant adapter_seed0 --steer-random-seed 1 --steer-positions from_last"),
    ("replace-random-s2", "__R__ --steer-constant adapter_seed0 --steer-random-seed 2 --steer-positions from_last"),
    ("replace-adapter-s0-all", "__R__ --steer-constant adapter_seed0 --steer-positions all"),
    ("replace-base-mean-all", "__R__ --steer-constant base_mean --steer-positions all"),
]

RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=__JOB__
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root, passing an absolute frozen snapshot path.
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
MODEL=__MODEL__
LAYER=__LAYER__
SUFFIX=__SUFFIX__
mkdir -p output/eval
if [[ ! -f output/constants.pt ]]; then
    echo "=== $(date -u +%FT%TZ) start constants"
    "$PY" -u make_constants.py --model "$MODEL" --revision __REVISION__ --layer "$LAYER" --pairs data/train_soo_pairs.jsonl \\
        __ADAPTER_SPECS__ --out output/constants.pt
    echo "=== $(date -u +%FT%TZ) done constants"
fi
R="--steer-constants output/constants.pt --steer-layer $LAYER --steer-mode replace"
run() {  # run NAME ORIENT ARGS...
    local name=$1 orient=$2; shift 2
    local data=data/eval; [[ "$orient" == mirrored ]] && data=data/eval_mirrored
    local tag="${name}_${orient}"
    if [[ -f "output/eval/${tag}_main_${SUFFIX}_summary.json" && -f "output/eval/${tag}_treasure_hunt_${SUFFIX}_summary.json" && -f "output/eval/${tag}_perspectives_${SUFFIX}_summary.json" ]]; then
        echo "skip $tag (complete)"; return
    fi
    echo "=== $(date -u +%FT%TZ) start $tag"
    "$PY" -u -m selfconcept.soo.evaluate --model "$MODEL" --data "$data" --out output/eval --tag "$tag" \\
        --scenarios main treasure_hunt perspectives --n 250 --suffix "$SUFFIX" "$@"
    echo "=== $(date -u +%FT%TZ) done $tag"
}
for orient in orig mirrored; do
__RUNS__
done
"$PY" analyze_constant_round.py output | tee output/analysis.log
echo "=== all runs complete"
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapters(spec):
    info = {}
    for seed in spec["seeds"]:
        d = SOO / spec["adapter_dir"] / f"seed{seed}"
        if not (d / "adapter_config.json").exists() or not (d / "adapter_model.safetensors").exists():
            raise SystemExit(f"incomplete adapter directory: {d}")
        log = json.loads((d / "train_log.json").read_text())
        cfg = log["config"]
        if cfg["layer"] != spec["layer"] or cfg["model"] != spec["model"] or log["seed"] != seed or cfg.get("soo_mode") != "last_token" or cfg["train_data"] != TRAIN_FILE:
            raise SystemExit(f"unexpected training record for {d}: {cfg}, seed {log['seed']}")
        info[f"seed{seed}"] = {"path": str(d.relative_to(ROOT)), "config": cfg, "final_epoch_loss": log["epoch_losses"][-1]}
    return info


def check_eval_data():
    info = {}
    for d in ("data/eval", "data/eval_mirrored"):
        for s in ("main", "treasure_hunt", "perspectives"):
            f = SOO / d / f"{s}.jsonl"
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if len(rows) < 250 or not all("honest_answer" in r and "deceptive_answer" in r for r in rows):
                raise SystemExit(f"{f}: need at least 250 rows with reference answers")
            info[f"{d}/{s}"] = {"rows": len(rows), "sha256": sha256(f)}
    return info


def freeze(snapshot, key, spec):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    shutil.copy2(SOO / TRAIN_FILE, snapshot / "data/train_soo_pairs.jsonl")
    for d in ("eval", "eval_mirrored"):
        (snapshot / "data" / d).mkdir()
        for s in ("main", "treasure_hunt", "perspectives"):
            shutil.copy2(SOO / "data" / d / f"{s}.jsonl", snapshot / "data" / d / f"{s}.jsonl")
    for seed in spec["seeds"]:
        shutil.copytree(SOO / spec["adapter_dir"] / f"seed{seed}", snapshot / "adapters" / f"seed{seed}")
    for name in ("CONSTANT_ROUND.md", "scripts/make_constants.py", "scripts/analyze_constant_round.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_constant_round.py")
    adapter_specs = " ".join(f"--adapter adapter_seed{s}=adapters/seed{s}" for s in spec["seeds"])
    runs = "\n".join(f'    run {name} "$orient" {args.replace("__R__", "$R")}'.rstrip() for name, args in CONDITIONS)
    script = (RUN_SH.replace("__JOB__", f"soo2-constants-{key}").replace("__HOURS__", f"{spec['hours']:02d}")
              .replace("__MODEL__", spec["model"]).replace("__REVISION__", spec["revision"]).replace("__LAYER__", str(spec["layer"]))
              .replace("__SUFFIX__", spec["suffix"]).replace("__ADAPTER_SPECS__", adapter_specs).replace("__RUNS__", runs))
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def software_checks(snapshot):
    env = {"PYTHONPATH": str(snapshot / "src"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s", str(snapshot / "tests"), "-p", "test_soo*.py", "-v"],
                          cwd=snapshot, env=env, text=True, capture_output=True)
    (snapshot / "software_checks.log").write_text(proc.stdout + proc.stderr)
    m = re.search(r"Ran (\d+) tests", proc.stderr + proc.stdout)
    if proc.returncode != 0 or not m:
        raise SystemExit(f"software checks failed; see {snapshot / 'software_checks.log'}")
    return f"{m.group(1)} tests passed on CPU; see software_checks.log"


def submit(snapshot, key):
    proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot)], cwd=ROOT, text=True, capture_output=True, check=True)
    job = proc.stdout.strip().split(";")[0]
    (snapshot / "submission.json").write_text(json.dumps({"job_id": job, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                                         "log": str(SOO / "slurm-logs" / f"soo2-constants-{key}-{job}.out")}, indent=2) + "\n")
    print(f"submitted job {job}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    key, spec = args.model_key, MODELS[args.model_key]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = SOO / "results/study2" / f"constants-{key}-L{spec['layer']}-{stamp}"
    adapters = check_adapters(spec)
    eval_data = check_eval_data()
    freeze(snapshot, key, spec)
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "CONSTANT_ROUND.md",
        "model_key": key, **spec, "conditions": [c for c, _ in CONDITIONS], "orientations": ["orig", "mirrored"],
        "scenarios": ["main", "treasure_hunt", "perspectives"], "n_per_scenario": 250,
        "adapters": adapters, "eval_data": eval_data, "test_result": test_result,
        "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        submit(snapshot, key)


if __name__ == "__main__":
    main()
