"""Freeze and submit the Kimi-Dev-72B training-layer sweep (KIMI_SWEEP_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_kimi_sweep.py [--no-submit]
One frozen snapshot, two jobs that split the layers. Per layer a job trains the seed-0 SOO LoRA at that layer, evaluates
main in both orientations, and, only if the pre-specified screen passes, the treasure hunt and perspectives. Base and the
L40 adapter cells are copied from the adapter round's snapshot.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_eval_data, sha256, software_checks
from launch_kimi_round import CONFIG, EVAL, check_model

JOBS = [[24, 56, 8, 72], [32, 48, 16, 64]]
PRIOR = "kimi-kimi-dev-72b-L40-20260929T233409Z"  # adapter round: base and the L40 adapter
PRIOR_LAYER = 40

RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-kimi-sweep
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root: run.sh SNAPSHOT "LAYER LAYER ...".
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
LAYERS=${2:?pass the layers for this job}
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit through Slurm from the repository root}
export PYTHONPATH="$SNAPSHOT_ROOT/src"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not assign visible GPUs; refusing to select unallocated GPUs."
    exit 1
fi
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; layers $LAYERS"
cd "$SNAPSHOT_ROOT"
PY="$REPO_ROOT/__VENV__/bin/python"
MODEL=__MODEL__
SUFFIX=room_only
mkdir -p output/eval
run() {  # run LAYER ORIENT SCENARIOS(comma-separated)
    local layer=$1 orient=$2 scen=$3
    local data=data/eval; [[ "$orient" == mirrored ]] && data=data/eval_mirrored
    local tag="L${layer}-adapter_${orient}"
    local s todo=""
    for s in ${scen//,/ }; do
        [[ -f "output/eval/${tag}_${s}_${SUFFIX}_summary.json" ]] || todo="$todo $s"
    done
    if [[ -z "$todo" ]]; then echo "skip $tag $scen (complete)"; return; fi
    echo "=== $(date -u +%FT%TZ) start $tag$todo"
    "$PY" -u -m selfconcept.soo.evaluate --model "$MODEL" --data "$data" --out output/eval --tag "$tag" \\
        --scenarios $todo --n 250 --suffix "$SUFFIX" __EVAL__ --adapter "output/checkpoints/kimi-dev-72b-L${layer}/seed0"
    echo "=== $(date -u +%FT%TZ) done $tag$todo"
}
for layer in $LAYERS; do
    if [[ ! -f output/checkpoints/kimi-dev-72b-L${layer}/seed0/adapter_model.safetensors ]]; then
        echo "=== $(date -u +%FT%TZ) start training L$layer"
        "$PY" -u -m selfconcept.soo.train --config kimi-dev-72b.yaml --layer "$layer" --seeds 0 --out output/checkpoints
        echo "=== $(date -u +%FT%TZ) done training L$layer"
    fi
    for orient in orig mirrored; do run "$layer" "$orient" main; done
    if "$PY" analyze_kimi_sweep.py output launch.json --screen "$layer" | tee -a output/screen.log; then
        for orient in orig mirrored; do run "$layer" "$orient" treasure_hunt,perspectives; done
    fi
done
"$PY" analyze_kimi_sweep.py output launch.json > output/analysis.log
echo "=== all runs complete"
"""


def freeze(snapshot, config, hours):
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
    for name in ("KIMI_SWEEP_ROUND.md", CONFIG, "scripts/analyze_layer_round.py", "scripts/analyze_kimi_sweep.py",
                 "scripts/make_constants.py", "scripts/extract_band_deltas.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_kimi_sweep.py")
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__HOURS__", f"{hours:02d}").replace("__MODEL__", config["model"])
              .replace("__EVAL__", EVAL))
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def carry_over(snapshot, prior):
    """Copy base and the prior round's full adapter (renamed L<layer>-adapter) cells; return the copied names."""
    out, names = snapshot / "output/eval", []
    out.mkdir(parents=True)
    for f in sorted((prior / "output/eval").iterdir()):
        for old, new in (("base_", "base_"), ("adapter-seed0_", f"L{PRIOR_LAYER}-adapter_")):
            if f.name.startswith(old):
                shutil.copy2(f, out / (new + f.name[len(old):]))
                names.append(new + f.name[len(old):])
    if sum(n.startswith("base_") and n.endswith("_summary.json") for n in names) != 6:
        raise SystemExit(f"expected six base summaries in {prior}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load((SOO / CONFIG).read_text())
    if config["train_data"] != TRAIN_FILE or config.get("soo_mode") != "last_token":
        raise SystemExit(f"unexpected training config: {config}")
    model_info = check_model(config)
    layers = [l for job in JOBS for l in job]
    if any(not 0 < l < model_info["num_hidden_layers"] for l in layers) or len(set(layers)) != len(layers) or PRIOR_LAYER in layers:
        raise SystemExit(f"bad layer list {layers}")
    results = (SOO / "results/study2").resolve()  # real path (results may be a symlink)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = results / f"kimisweep-kimi-dev-72b-{stamp}"
    eval_data = check_eval_data()
    freeze(snapshot, config, args.hours)
    test_result = software_checks(snapshot)
    carried = carry_over(snapshot, results / PRIOR)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "KIMI_SWEEP_ROUND.md", "round": "kimisweep",
        "python": str(VENV / "bin/python"), "model_key": "kimi-dev-72b", "model": config["model"], "revision": config["revision"],
        "layers": sorted(layers), "jobs": JOBS, "carried_layers": [PRIOR_LAYER], "carried_from": PRIOR, "carried_over": carried,
        "suffix": "room_only", "training_config": config, "model_check": model_info, "eval_args": EVAL,
        "orientations": ["orig", "mirrored"], "scenarios": ["main", "treasure_hunt", "perspectives"], "n_per_scenario": 250,
        "eval_data": eval_data, "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        jobs = []
        for job_layers in JOBS:
            proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot), " ".join(map(str, job_layers))],
                                  cwd=ROOT, text=True, capture_output=True, check=True)
            job = proc.stdout.strip().split(";")[0]
            jobs.append({"job_id": job, "layers": job_layers, "log": str(SOO / "slurm-logs" / f"soo2-kimi-sweep-{job}.out")})
            print(f"submitted job {job} for layers {job_layers}")
        (snapshot / "submission.json").write_text(json.dumps({"jobs": jobs, "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
