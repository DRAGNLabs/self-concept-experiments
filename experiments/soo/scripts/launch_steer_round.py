"""Freeze and submit the band-steering round for one model (STEER_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_steer_round.py --model-key gemma4-12b [--no-submit]
The job measures the band's mean LoRA deltas on the training prompts (extract_band_deltas.py), evaluates the
references and every steered cell in both orientations on main / treasure_hunt / perspectives (n=250), measures
the self/other and name-control gaps under each intervention (measure_overlap), runs the intent split on
Mistral, then the pre-specified analysis.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_adapter, check_eval_data, sha256, software_checks

PROBES = "data/subspace_pilot_pairs.jsonl"

MODELS = {
    "gemma4-12b": dict(model="google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", layer=24,
                       adapter_dir="results/checkpoints/gemma4-12b-L24", suffix="room_only", hours=8,
                       band=(19, 23), bisect="band (19-23): main D 0/0, TH 0/0, perspectives 100/100, no refusals"),
    "mistral-7b": dict(model="mistralai/Mistral-7B-Instruct-v0.2", revision="63a8b081895390a26e140280378bc85ec8bce07a", layer=16,
                       adapter_dir="results/checkpoints/mistral-7b-lasttok-L16", suffix="i_would", hours=30,
                       band=(5, 13), bisect="band (5-13): main D 13.6/0, clean honest 75/91%"),
}

BAND = "--band-deltas output/band_deltas.pt"


def conditions(spec):
    a, b = spec["band"]
    band_lora = f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules v_proj"
    v_resp = f"{BAND} --band-module v_proj --band-token-mode last --steer-positions from_last"
    v_all = f"{BAND} --band-module v_proj --band-token-mode all --steer-positions all"
    q_resp = f"{BAND} --band-module q_proj --band-token-mode last --steer-positions from_last"
    q_all = f"{BAND} --band-module q_proj --band-token-mode all --steer-positions all"
    return [("base", ""), ("adapter-seed0", "--adapter adapters/seed0"), ("band-v", band_lora),
            ("v-resp-a1", f"{v_resp} --band-alpha 1"), ("v-resp-a2", f"{v_resp} --band-alpha 2"),
            ("v-all-a1", f"{v_all} --band-alpha 1"), ("v-all-a2", f"{v_all} --band-alpha 2"),
            ("rand-resp-a1", f"{v_resp} --band-alpha 1 --band-random-seed 0"), ("rand-resp-a2", f"{v_resp} --band-alpha 2 --band-random-seed 0"),
            ("rand-all-a1", f"{v_all} --band-alpha 1 --band-random-seed 0"), ("rand-all-a2", f"{v_all} --band-alpha 2 --band-random-seed 0"),
            ("q-resp-a1", f"{q_resp} --band-alpha 1"), ("q-resp-a2", f"{q_resp} --band-alpha 2"), ("q-all-a1", f"{q_all} --band-alpha 1")]


def overlap_runs(spec):
    a, b = spec["band"]
    subset = f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules v_proj"
    band = f"{BAND} --band-modules v_proj q_proj --band-random-seeds 0 {subset}"
    return [("resp-a1", f"{band} --positions from_last --band-token-mode last --band-alpha 1"),
            ("resp-a2", f"{band} --positions from_last --band-token-mode last --band-alpha 2"),
            ("all-a1", f"{band} --positions all --band-token-mode all --band-alpha 1"),
            ("all-a2", f"{band} --positions all --band-token-mode all --band-alpha 2"),
            ("full-adapter", "--adapter adapters/seed0")]


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
# byutils moves HF_HOME to an empty autodelete cache unless it is already set; the models live in the default cache.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
# This cluster's system OpenSSL config fails its FIPS self-test; the login profile sets this too.
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
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
PY="$REPO_ROOT/__VENV__/bin/python"
MODEL=__MODEL__
REVISION=__REVISION__
LAYER=__LAYER__
SUFFIX=__SUFFIX__
mkdir -p output/eval output/overlap
if [[ ! -f output/band_deltas.pt ]]; then
    echo "=== $(date -u +%FT%TZ) start band delta extraction"
    "$PY" -u extract_band_deltas.py --model "$MODEL" --revision "$REVISION" --adapter adapters/seed0 --layers __BAND_SPEC__ \\
        --modules q_proj,v_proj --pairs data/train_soo_pairs.jsonl --out output/band_deltas.pt
    echo "=== $(date -u +%FT%TZ) done band delta extraction"
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
        --scenarios ${scen//,/ } --n 250 --suffix "$SUFFIX" "$@"
    echo "=== $(date -u +%FT%TZ) done $tag"
}
overlap() {  # overlap NAME ARGS...
    local name=$1; shift
    local out="output/overlap/$name"
    if [[ -f "$out/manifest.json" ]]; then echo "skip overlap $name (complete)"; return; fi
    rm -rf "$out"
    echo "=== $(date -u +%FT%TZ) start overlap $name"
    "$PY" -u -m selfconcept.soo.measure_overlap --model "$MODEL" --revision "$REVISION" --probes __PROBES__ --split development \\
        --layer __BAND_TOP__ --residual-layers __BAND_TOP__ "$LAYER" __LAST_LAYER__ --device cuda --dtype bfloat16 --batch-size 4 \\
        --bootstrap 2000 "$@" --out "$out"
    echo "=== $(date -u +%FT%TZ) done overlap $name"
}
ALL=main,treasure_hunt,perspectives
for orient in orig mirrored; do
__RUNS__
done
__OVERLAPS__
if [[ "$SUFFIX" == i_would ]]; then
    "$PY" intent_language_check.py . --suffix "$SUFFIX" | tee output/intent_split.log
fi
"$PY" analyze_steer_round.py output launch.json | tee output/analysis.log
echo "=== all runs complete"
"""


def freeze(snapshot, key, spec, n_layers):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    shutil.copy2(SOO / TRAIN_FILE, snapshot / "data/train_soo_pairs.jsonl")
    shutil.copy2(SOO / PROBES, snapshot / PROBES)
    for d in ("eval", "eval_mirrored"):
        (snapshot / "data" / d).mkdir()
        for s in ("main", "treasure_hunt", "perspectives"):
            shutil.copy2(SOO / "data" / d / f"{s}.jsonl", snapshot / "data" / d / f"{s}.jsonl")
    shutil.copytree(SOO / spec["adapter_dir"] / "seed0", snapshot / "adapters" / "seed0")
    for name in ("STEER_ROUND.md", "scripts/make_constants.py", "scripts/extract_band_deltas.py", "scripts/analyze_layer_round.py",
                 "scripts/analyze_steer_round.py", "scripts/intent_language_check.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_steer_round.py")
    a, b = spec["band"]
    runs = "\n".join(f'    run {name} "$orient" "$ALL" {args}'.rstrip() for name, args in conditions(spec))
    overlaps = "\n".join(f"overlap {name} {args}" for name, args in overlap_runs(spec))
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__JOB__", f"soo2-steer-{key}").replace("__HOURS__", f"{spec['hours']:02d}")
              .replace("__MODEL__", spec["model"]).replace("__REVISION__", spec["revision"]).replace("__LAYER__", str(spec["layer"]))
              .replace("__SUFFIX__", spec["suffix"]).replace("__BAND_SPEC__", f"range:{a}-{b}").replace("__BAND_TOP__", str(b))
              .replace("__LAST_LAYER__", str(n_layers - 1)).replace("__PROBES__", PROBES)
              .replace("__RUNS__", runs).replace("__OVERLAPS__", overlaps))
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def submit(snapshot, key):
    proc = subprocess.run(["sbatch", "--parsable", str(snapshot / "run.sh"), str(snapshot)], cwd=ROOT, text=True, capture_output=True, check=True)
    job = proc.stdout.strip().split(";")[0]
    (snapshot / "submission.json").write_text(json.dumps({"job_id": job, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                                         "log": str(SOO / "slurm-logs" / f"soo2-steer-{key}-{job}.out")}, indent=2) + "\n")
    print(f"submitted job {job}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    key, spec = args.model_key, MODELS[args.model_key]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = SOO / "results/study2" / f"steer-{key}-L{spec['layer']}-{stamp}"
    adapter = check_adapter(spec)
    eval_data = check_eval_data()
    if spec["band"][1] >= spec["layer"]:
        raise SystemExit("the band must lie below the training layer")
    freeze(snapshot, key, spec, adapter["n_layers"])
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "STEER_ROUND.md",
        "python": str(VENV / "bin/python"), "model_key": key, **{k: v for k, v in spec.items() if k != "band"},
        "band_layers": list(range(spec["band"][0], spec["band"][1] + 1)), "n_layers": adapter["n_layers"],
        "conditions": dict(conditions(spec)), "overlap_runs": dict(overlap_runs(spec)),
        "orientations": ["orig", "mirrored"], "scenarios": ["main", "treasure_hunt", "perspectives"], "n_per_scenario": 250,
        "probes": PROBES, "adapter": adapter, "eval_data": eval_data, "test_result": test_result,
        "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        submit(snapshot, key)


if __name__ == "__main__":
    main()
