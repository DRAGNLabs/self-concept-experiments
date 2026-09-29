"""Freeze and submit the Kimi-Dev-72B adapter round (KIMI_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_kimi_round.py [--no-submit]
The job trains the seed-0 SOO LoRA at L40 (configs/kimi-dev-72b.yaml, the Gemma-2 paper recipe) inside the
snapshot, evaluates the base model and the adapter in both orientations on main / treasure_hunt / perspectives
(n=250, 1024 new tokens, classification after the think block, batched greedy decoding), and stops there if the
adapter's main-scenario effect is null under the pre-specified rule. Otherwise it runs the layer round's coarse
subsets (training layer only, below it, v_proj only, q_proj only) and the analysis.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_eval_data, sha256, software_checks

CONFIG = "configs/kimi-dev-72b.yaml"
THINK = "--think-open ◁think▷ --think-close ◁/think▷"
EVAL = f"--device-map auto --batch-size 16 --max-new-tokens 1024 {THINK}"
ADAPTER = "output/checkpoints/kimi-dev-72b/seed0"


def conditions(layer):
    """Gate conditions first; the coarse subsets run only if the adapter is not null."""
    gate = [("base", ""), ("adapter-seed0", f"--adapter {ADAPTER}")]
    coarse = [("only-train", f"--adapter {ADAPTER} --adapter-layers only:{layer}"),
              ("below-train", f"--adapter {ADAPTER} --adapter-layers below:{layer}"),
              ("v-only", f"--adapter {ADAPTER} --adapter-modules v_proj"),
              ("q-only", f"--adapter {ADAPTER} --adapter-modules q_proj")]
    return gate, coarse


RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-kimi-dev-72b
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root, passing an absolute frozen snapshot path.
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
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
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
cd "$SNAPSHOT_ROOT"
PY="$REPO_ROOT/__VENV__/bin/python"
MODEL=__MODEL__
SUFFIX=room_only
mkdir -p output/eval
if [[ ! -f __ADAPTER__/adapter_model.safetensors ]]; then
    echo "=== $(date -u +%FT%TZ) start training (seed 0, L__LAYER__)"
    "$PY" -u -m selfconcept.soo.train --config kimi-dev-72b.yaml --seeds 0 --out output/checkpoints
    echo "=== $(date -u +%FT%TZ) done training"
fi
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
ALL=main,treasure_hunt,perspectives
for orient in orig mirrored; do
__GATE_RUNS__
done
if ! "$PY" analyze_kimi_round.py output launch.json --gate | tee output/gate.log; then
    echo "=== adapter null under the pre-specified rule; stopping before the coarse subsets"
    "$PY" analyze_kimi_round.py output launch.json | tee output/analysis.log
    exit 0
fi
for orient in orig mirrored; do
__COARSE_RUNS__
done
"$PY" analyze_kimi_round.py output launch.json | tee output/analysis.log
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
    for name in ("KIMI_ROUND.md", CONFIG, "scripts/analyze_layer_round.py", "scripts/analyze_kimi_round.py",
                 "scripts/make_constants.py", "scripts/extract_band_deltas.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_kimi_round.py")
    gate, coarse = conditions(config["layer"])
    fmt = lambda conds: "\n".join(f'    run {name} "$orient" "$ALL" {args}'.rstrip() for name, args in conds)  # noqa: E731
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__HOURS__", f"{hours:02d}").replace("__MODEL__", config["model"])
              .replace("__LAYER__", str(config["layer"])).replace("__ADAPTER__", ADAPTER).replace("__EVAL__", EVAL)
              .replace("__GATE_RUNS__", fmt(gate)).replace("__COARSE_RUNS__", fmt(coarse)))
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def check_model(config):
    snap = Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-Dev-72B/snapshots" / config["revision"]
    shards = sorted(snap.glob("*.safetensors"))
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    needed = sorted(set(index["weight_map"].values()))
    if [p.name for p in shards] != needed:
        raise SystemExit(f"Kimi-Dev-72B snapshot {config['revision']} is incomplete: {len(shards)} of {len(needed)} shards")
    cfg = json.loads((snap / "config.json").read_text())
    return {"revision": config["revision"], "n_shards": len(shards), "architectures": cfg.get("architectures"),
            "num_hidden_layers": cfg.get("num_hidden_layers"), "hidden_size": cfg.get("hidden_size")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load((SOO / CONFIG).read_text())
    if config["train_data"] != TRAIN_FILE or config.get("soo_mode") != "last_token":
        raise SystemExit(f"unexpected training config: {config}")
    model_info = check_model(config)
    if not 0 < config["layer"] < model_info["num_hidden_layers"]:
        raise SystemExit("training layer out of range")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = (SOO / "results/study2").resolve() / f"kimi-kimi-dev-72b-L{config['layer']}-{stamp}"  # real path (results may be a symlink)
    eval_data = check_eval_data()
    freeze(snapshot, config, args.hours)
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    gate, coarse = conditions(config["layer"])
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "KIMI_ROUND.md", "round": "kimi",
        "python": str(VENV / "bin/python"), "model_key": "kimi-dev-72b", "model": config["model"], "revision": config["revision"],
        "layer": config["layer"], "suffix": "room_only", "training_config": config, "model_check": model_info,
        "eval_args": EVAL, "conditions": dict(gate + coarse), "gate_conditions": [c for c, _ in gate],
        "orientations": ["orig", "mirrored"], "scenarios": ["main", "treasure_hunt", "perspectives"], "n_per_scenario": 250,
        "eval_data": eval_data, "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot)], cwd=ROOT, text=True, capture_output=True, check=True)
        job = proc.stdout.strip().split(";")[0]
        (snapshot / "submission.json").write_text(json.dumps({"job_id": job, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                                             "log": str(SOO / "slurm-logs" / f"soo2-kimi-dev-72b-{job}.out")}, indent=2) + "\n")
        print(f"submitted job {job}")


if __name__ == "__main__":
    main()
