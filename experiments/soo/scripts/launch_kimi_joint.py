"""Freeze and submit the Kimi-Dev-72B L56 joint-offset round (KIMI_JOINT_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_kimi_joint.py [--no-submit]
One frozen snapshot, three jobs (A, B, C) that split the fifteen offset cells. No adapter is loaded: every cell steers
the base model with the mean LoRA deltas measured in the localization round (its band_deltas.pt is copied in), and the
reference cells (base, the adapter, the LoRA subsets, the alpha = 1 steering cells) are carried over from that snapshot.
Every cell is main + perspectives in both orientations (n=250, 1024 new tokens, classification after the think block).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import torch

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_eval_data, sha256, software_checks
from launch_kimi_round import CONFIG, EVAL, check_model

LAYER = 56
PRIOR = "kimiloc-kimi-dev-72b-L56-20260930T223131Z"  # localization round: band deltas and the reference cells
SPEC = dict(model="moonshotai/Kimi-Dev-72B", revision="8791d7981945752a51f692d66f2bbfb3573c9722", layer=LAYER, suffix="room_only")
SCENARIOS = "main,perspectives"
JOINT = "q_proj,v_proj"
BANDS = {"all": ("range:0-55", "0-55"), "hi": ("range:28-55", "28-55")}  # band -> (layer spec, label)
LORA = {("all", JOINT): "below-train", ("hi", JOINT): "hi-half", ("all", "v_proj"): "below-v"}  # the LoRA an offset replaces
CARRIED = ("base", "adapter-seed0", "below-train", "hi-half", "below-v", "below-q", "v-all-a1", "rand-all-a1", "q-all-a1")


def cell(band, alpha, positions="all", module=JOINT, random=False):
    return {"band": band, "layers": BANDS[band][1], "module": module, "positions": positions, "alpha": alpha, "random": random,
            "reference": LORA[(band, module)]}


CELLS = {
    "jv-all-a0.25": cell("all", 0.25), "jv-all-a0.5": cell("all", 0.5), "jv-all-a0.75": cell("all", 0.75),
    "rand-all-a0.25": cell("all", 0.25, random=True), "rand-all-a0.5": cell("all", 0.5, random=True),
    "rand-all-a0.75": cell("all", 0.75, random=True),
    "jv-hi-a0.5": cell("hi", 0.5), "jv-hi-a1": cell("hi", 1), "rand-hi-a0.5": cell("hi", 0.5, random=True),
    "rand-hi-a1": cell("hi", 1, random=True),
    "jv-all-content-a1": cell("all", 1, "content"), "rand-all-content-a1": cell("all", 1, "content", random=True),
    "jv-hi-content-a1": cell("hi", 1, "content"), "rand-hi-content-a1": cell("hi", 1, "content", random=True),
    "v-all-a0.5": cell("all", 0.5, module="v_proj"),
}
for _name, _c in CELLS.items():  # a real joint offset's matched random: same band, positions and alpha
    _c["control"] = None if _c["random"] else next(
        (n for n, r in CELLS.items() if r["random"] and all(r[k] == _c[k] for k in ("band", "module", "positions", "alpha"))), None)
JOBS = {"A": ["jv-all-a0.5", "rand-all-a0.5", "jv-all-a0.25", "rand-all-a0.25", "v-all-a0.5"],
        "B": ["jv-hi-a1", "rand-hi-a1", "jv-hi-a0.5", "rand-hi-a0.5", "jv-all-a0.75"],
        "C": ["jv-all-content-a1", "rand-all-content-a1", "jv-hi-content-a1", "rand-hi-content-a1", "rand-all-a0.75"]}


def condition_args(name):
    c = CELLS[name]
    return (f"--band-deltas output/band_deltas.pt --band-token-mode all --band-module {c['module']} --band-layers {BANDS[c['band']][0]} "
            f"--steer-positions {c['positions']} --band-alpha {c['alpha']}" + (" --band-random-seed 0" if c["random"] else ""))


RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-kimi-joint
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root: run.sh SNAPSHOT A|B|C.
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
JOB=${2:?pass the job letter (A, B or C)}
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
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; job $JOB"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
cd "$SNAPSHOT_ROOT"
PY="$REPO_ROOT/__VENV__/bin/python"
MODEL=__MODEL__
SUFFIX=room_only
run() {  # run NAME ORIENT SCENARIOS(comma-separated) ARGS...
    local name=$1 orient=$2 scen=$3; shift 3
    local data=data/eval; [[ "$orient" == mirrored ]] && data=data/eval_mirrored
    local tag="${name}_${orient}"
    local s complete=1
    for s in ${scen//,/ }; do
        [[ -f "output/eval/${tag}_${s}_${SUFFIX}_summary.json" ]] || complete=0
    done
    if [[ $complete == 1 ]]; then echo "skip $tag (complete)"; return; fi
    echo "=== $(date -u +%FT%TZ) start $tag"
    "$PY" -u -m selfconcept.soo.evaluate --model "$MODEL" --data "$data" --out output/eval --tag "$tag" \\
        --scenarios ${scen//,/ } --n 250 --suffix "$SUFFIX" __EVAL__ "$@"
    echo "=== $(date -u +%FT%TZ) done $tag"
}
case "$JOB" in
A)
__RUNS_A__
    ;;
B)
__RUNS_B__
    ;;
C)
__RUNS_C__
    ;;
*) echo "unknown job $JOB"; exit 1 ;;
esac
"$PY" analyze_kimi_joint.py output launch.json | tee "output/analysis_$JOB.log"
echo "=== job $JOB complete"
"""


def check_deltas(path):
    """The localization round's band deltas must hold an 'all'-token mean for q_proj and v_proj in every layer 0-55."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    want = {f"{l}:{m}" for l in range(LAYER) for m in JOINT.split(",")}
    if data.get("model") != SPEC["model"] or data.get("revision") != SPEC["revision"] or set(data["offsets"]["all"]) != want:
        raise SystemExit(f"{path} is not the L{LAYER} adapter's 0-{LAYER - 1} q_proj/v_proj deltas for {SPEC['model']}")
    norm = lambda keys: round(float(sum(data["offsets"]["all"][k].float().norm() ** 2 for k in keys) ** 0.5), 4)  # noqa: E731
    return {"source": str(path), "sha256": sha256(path), "keys": len(want), "adapter": data.get("adapter"), "n_prompts": data.get("n_prompts"),
            "root_sum_square_norm": {f"{band}:{m}": norm([f"{l}:{m}" for l in range(int(lab.split('-')[0]), LAYER)])
                                     for band, (_, lab) in BANDS.items() for m in JOINT.split(",")}}


def freeze(snapshot, hours, prior):
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
    for name in ("KIMI_JOINT_ROUND.md", CONFIG, "scripts/analyze_layer_round.py", "scripts/analyze_kimi_loc.py", "scripts/analyze_kimi_joint.py",
                 "scripts/make_constants.py", "scripts/extract_band_deltas.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_kimi_joint.py")
    (snapshot / "output/eval").mkdir(parents=True)
    for ext in ("pt", "json"):
        shutil.copy2(prior / f"output/band_deltas.{ext}", snapshot / f"output/band_deltas.{ext}")
    runs = {j: "\n".join(f'    for orient in orig mirrored; do run {n} "$orient" "{SCENARIOS}" {condition_args(n)}; done'
                         for n in names) for j, names in JOBS.items()}
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__HOURS__", f"{hours:02d}").replace("__MODEL__", SPEC["model"])
              .replace("__EVAL__", EVAL))
    for j in JOBS:
        script = script.replace(f"__RUNS_{j}__", runs[j])
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def carry_over(snapshot, prior):
    """Copy the reference cells' main and perspectives files from the localization snapshot; return the copied names."""
    out, names = snapshot / "output/eval", []
    for cond in CARRIED:
        found = 0
        for orient in ("orig", "mirrored"):
            for scen in SCENARIOS.split(","):
                stem = f"{cond}_{orient}_{scen}_{SPEC['suffix']}"
                for tail in ("_summary.json", ".jsonl"):
                    src = prior / "output/eval" / (stem + tail)
                    if not src.exists():
                        raise SystemExit(f"missing {src}")
                    shutil.copy2(src, out / src.name)
                    names.append(src.name)
                found += 1
        assert found == 4
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    model_info = check_model(SPEC)
    if sorted(n for names in JOBS.values() for n in names) != sorted(CELLS):
        raise SystemExit("jobs must partition the cells")
    if any(not c["random"] and c["module"] == JOINT and c["control"] is None for c in CELLS.values()):
        raise SystemExit("every joint offset needs a matched random cell")
    results = (SOO / "results/study2").resolve()  # real path (results may be a symlink)
    prior = results / PRIOR
    deltas = check_deltas(prior / "output/band_deltas.pt")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = results / f"kimijoint-kimi-dev-72b-L{LAYER}-{stamp}"
    eval_data = check_eval_data()
    freeze(snapshot, args.hours, prior)
    test_result = software_checks(snapshot)
    carried = carry_over(snapshot, prior)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    conditions = {**{c: "(carried over)" for c in CARRIED}, **{n: condition_args(n) for names in JOBS.values() for n in names}}
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "KIMI_JOINT_ROUND.md", "round": "kimijoint",
        "python": str(VENV / "bin/python"), "model_key": "kimi-dev-72b", **SPEC,
        "conditions": conditions, "cells": CELLS, "jobs": JOBS, "carried_from": PRIOR, "carried_over": carried, "band_deltas": deltas,
        "model_check": model_info, "eval_args": EVAL,
        "orientations": ["orig", "mirrored"], "scenarios": SCENARIOS.split(","), "n_per_scenario": 250,
        "eval_data": eval_data, "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        jobs = []
        for letter in JOBS:
            proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot), letter],
                                  cwd=ROOT, text=True, capture_output=True, check=True)
            job = proc.stdout.strip().split(";")[0]
            jobs.append({"job_id": job, "job": letter, "conditions": JOBS[letter], "log": str(SOO / "slurm-logs" / f"soo2-kimi-joint-{job}.out")})
            print(f"submitted job {job} ({letter}: {', '.join(JOBS[letter])})")
        (snapshot / "submission.json").write_text(json.dumps({"jobs": jobs, "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
