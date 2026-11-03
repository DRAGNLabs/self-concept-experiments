"""Freeze and submit the Kimi-Dev-72B L56 localization round (KIMI_LOC_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_kimi_loc.py [--no-submit]
One frozen snapshot, two jobs (A, B) that split the conditions. The sweep's L56 adapter is copied into the snapshot;
its base and full-adapter cells are carried over. Job A measures the below-band's mean LoRA deltas (extract_band_deltas.py,
sharded) and runs the three steering cells before its subsets; job B runs the other subsets. Every cell is main +
perspectives in both orientations (n=250, 1024 new tokens, classification after the think block).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_adapter, check_eval_data, sha256, software_checks
from launch_kimi_round import CONFIG, EVAL, check_model

LAYER = 56
SWEEP = "kimisweep-kimi-dev-72b-20260930T171237Z"  # training-layer sweep: the L56 adapter, base and L56-adapter cells
SPEC = dict(model="moonshotai/Kimi-Dev-72B", revision="8791d7981945752a51f692d66f2bbfb3573c9722", layer=LAYER,
            adapter_dir=f"results/study2/{SWEEP}/output/checkpoints/kimi-dev-72b-L{LAYER}", suffix="room_only")
BAND = "--band-deltas output/band_deltas.pt --band-token-mode all --steer-positions all --band-alpha 1"
SCENARIOS = "main,perspectives"

# name -> (layer spec, module spec); bisection ranges are inclusive
SUBSETS = {
    "only-train": (f"only:{LAYER}", "q_proj,v_proj"), "below-train": (f"below:{LAYER}", "q_proj,v_proj"),
    "below-v": (f"below:{LAYER}", "v_proj"), "below-q": (f"below:{LAYER}", "q_proj"),
    "lo-half": ("range:0-27", "q_proj,v_proj"), "hi-half": ("range:28-55", "q_proj,v_proj"),
    "q1": ("range:0-13", "q_proj,v_proj"), "q2": ("range:14-27", "q_proj,v_proj"),
    "q3": ("range:28-41", "q_proj,v_proj"), "q4": ("range:42-55", "q_proj,v_proj"),
    "top8": ("range:48-55", "q_proj,v_proj"), "top4": ("range:52-55", "q_proj,v_proj"),
}
STEER = {"v-all-a1": f"{BAND} --band-module v_proj", "rand-all-a1": f"{BAND} --band-module v_proj --band-random-seed 0",
         "q-all-a1": f"{BAND} --band-module q_proj"}
JOBS = {"A": ["v-all-a1", "rand-all-a1", "q-all-a1", "below-v", "only-train", "hi-half", "q4"],
        "B": ["below-q", "below-train", "lo-half", "q3", "q1", "q2", "top8", "top4"]}


def condition_args(name):
    if name in STEER:
        return STEER[name]
    layers, modules = SUBSETS[name]
    return f"--adapter adapters/seed0 --adapter-layers {layers} --adapter-modules {modules}"


RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-kimi-loc
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root: run.sh SNAPSHOT A|B.
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
JOB=${2:?pass the job letter (A or B)}
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
REVISION=__REVISION__
SUFFIX=room_only
mkdir -p output/eval
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
if [[ "$JOB" == A && ! -f output/band_deltas.pt ]]; then
    echo "=== $(date -u +%FT%TZ) start band delta extraction (layers 0-__BELOW__)"
    "$PY" -u extract_band_deltas.py --model "$MODEL" --revision "$REVISION" --adapter adapters/seed0 --layers below:__LAYER__ \\
        --modules q_proj,v_proj --pairs data/train_soo_pairs.jsonl --out output/band_deltas.pt --device-map auto
    echo "=== $(date -u +%FT%TZ) done band delta extraction"
fi
case "$JOB" in
A)
__RUNS_A__
    ;;
B)
__RUNS_B__
    ;;
*) echo "unknown job $JOB"; exit 1 ;;
esac
"$PY" analyze_kimi_loc.py output launch.json | tee "output/analysis_$JOB.log"
echo "=== job $JOB complete"
"""


def freeze(snapshot, hours):
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
    shutil.copytree(SOO / SPEC["adapter_dir"] / "seed0", snapshot / "adapters" / "seed0")
    for name in ("KIMI_LOC_ROUND.md", CONFIG, "scripts/analyze_layer_round.py", "scripts/analyze_kimi_loc.py",
                 "scripts/make_constants.py", "scripts/extract_band_deltas.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_kimi_loc.py")
    runs = {j: "\n".join(f'    for orient in orig mirrored; do run {n} "$orient" "{SCENARIOS}" {condition_args(n)}; done'
                         for n in names) for j, names in JOBS.items()}
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__HOURS__", f"{hours:02d}").replace("__MODEL__", SPEC["model"])
              .replace("__REVISION__", SPEC["revision"]).replace("__EVAL__", EVAL).replace("__LAYER__", str(LAYER))
              .replace("__BELOW__", str(LAYER - 1)).replace("__RUNS_A__", runs["A"]).replace("__RUNS_B__", runs["B"]))
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def carry_over(snapshot, prior):
    """Copy base and the sweep's L56-adapter cells (renamed adapter-seed0); return the copied names."""
    out, names = snapshot / "output/eval", []
    out.mkdir(parents=True)
    for f in sorted((prior / "output/eval").iterdir()):
        for old, new in (("base_", "base_"), (f"L{LAYER}-adapter_", "adapter-seed0_")):
            if f.name.startswith(old):
                shutil.copy2(f, out / (new + f.name[len(old):]))
                names.append(new + f.name[len(old):])
    for prefix in ("base_", "adapter-seed0_"):
        if sum(n.startswith(prefix) and n.endswith("_summary.json") for n in names) != 6:
            raise SystemExit(f"expected six {prefix} summaries in {prior}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    model_info = check_model(SPEC)
    adapter = check_adapter(SPEC)
    if adapter["layers_with_nonzero_B"] != list(range(LAYER + 1)):
        raise SystemExit(f"expected nonzero LoRA B exactly at layers 0-{LAYER}: {adapter['layers_with_nonzero_B']}")
    if set(JOBS["A"]) | set(JOBS["B"]) != set(SUBSETS) | set(STEER) or len(JOBS["A"]) + len(JOBS["B"]) != len(SUBSETS) + len(STEER):
        raise SystemExit("jobs must partition the conditions")
    results = (SOO / "results/study2").resolve()  # real path (results may be a symlink)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = results / f"kimiloc-kimi-dev-72b-L{LAYER}-{stamp}"
    eval_data = check_eval_data()
    freeze(snapshot, args.hours)
    test_result = software_checks(snapshot)
    carried = carry_over(snapshot, results / SWEEP)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    conditions = {"base": "", "adapter-seed0": "--adapter adapters/seed0", **{n: condition_args(n) for n in SUBSETS}, **STEER}
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "KIMI_LOC_ROUND.md", "round": "kimiloc",
        "python": str(VENV / "bin/python"), "model_key": "kimi-dev-72b", **SPEC, "band_layers": list(range(LAYER)),
        "conditions": conditions, "subsets": {n: list(v) for n, v in SUBSETS.items()}, "steer_conditions": list(STEER),
        "steer_reference": "below-v", "jobs": JOBS, "carried_from": SWEEP, "carried_over": carried,
        "model_check": model_info, "adapter": adapter, "eval_args": EVAL,
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
            jobs.append({"job_id": job, "job": letter, "conditions": JOBS[letter], "log": str(SOO / "slurm-logs" / f"soo2-kimi-loc-{job}.out")})
            print(f"submitted job {job} ({letter}: {', '.join(JOBS[letter])})")
        (snapshot / "submission.json").write_text(json.dumps({"jobs": jobs, "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
