"""Freeze and submit the held-out round for one model (HELDOUT_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_heldout_round.py --model-key KEY --selection DIR [--no-submit]
      [--partition P --qos Q --gres G]
KEY is one of mistral-7b, gemma4-12b, gemma4-31b, qwen38-27b, kimi-dev-72b; DIR the heldout-selection-* directory written
by select_heldout.py. One frozen snapshot per model with the held-out eval sets (data/eval_heldout*), the seed-0 adapter,
the development band deltas (and constants) copied in, and run.sh SNAPSHOT JOB for each of the model's jobs. All jobs
write into the same snapshot; the readout (analyze_heldout_round.py) runs once every job has finished.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from launch_bisect_round import ROOT, SOO, TRAIN_FILE, VENV, check_adapter, sha256, software_checks
from launch_steer_round import MODELS as STEER_MODELS

STUDY2 = (SOO / "results/study2").resolve()
VARIANTS = ["name", "objective", "action", "name_objective", "name_action", "objective_action", "name_objective_action", "escape_room"]
MAIN3 = "main,treasure_hunt,perspectives"
KIMI_SPEC = dict(model="moonshotai/Kimi-Dev-72B", revision="8791d7981945752a51f692d66f2bbfb3573c9722", layer=56, suffix="room_only",
                 adapter_dir="results/study2/kimisweep-kimi-dev-72b-20260930T171237Z/output/checkpoints/kimi-dev-72b-L56",
                 modules=["q_proj", "v_proj"])
KIMI_EVAL = "--device-map auto --batch-size 16 --max-new-tokens 1024 --think-open ◁think▷ --think-close ◁/think▷"
# Band deltas: the band-steering round's extraction (identical to the dose round's; the control round's H100 re-extraction
# differs by at most 0.8% relative). Kimi: the localization round's, as used by the joint-offset and control rounds.
DELTAS = {"mistral-7b": "steer-mistral-7b-L16-20260928T230233Z", "gemma4-12b": "steer-gemma4-12b-L24-20260928T230232Z",
          "gemma4-31b": "steer-gemma4-31b-L32-20260929T180338Z", "qwen38-27b": "steer-qwen38-27b-L32-20260929T221532Z",
          "kimi-dev-72b": "kimijoint-kimi-dev-72b-L56-20261001T202619Z"}
CONSTANTS = {"mistral-7b": "constants-mistral-7b-L16-20260922T175427Z", "gemma4-12b": "constants-gemma4-12b-L24-20260922T175333Z",
             "gemma4-31b": "constants-gemma4-31b-L32-20260922T175236Z"}
HOURS = {"mistral-7b": 12, "gemma4-12b": 8, "gemma4-31b": 8, "qwen38-27b": 8, "kimi-dev-72b": 6}
KIMI_BANDS = {"all": "range:0-55", "hi": "range:28-55"}
KIMI_REF = {"all": "below-train", "hi": "hi-half"}


def offset_args(key, sel_or_alpha, positions="all", seed=None, band=None):
    alpha = sel_or_alpha
    if key == "kimi-dev-72b":
        return (f"--band-deltas output/band_deltas.pt --band-token-mode all --band-module q_proj,v_proj --band-layers {KIMI_BANDS[band]} "
                f"--steer-positions {positions} --band-alpha {alpha}" + (f" --band-random-seed {seed}" if seed is not None else ""))
    module = "in_proj_qkv" if key == "qwen38-27b" else "v_proj"
    return (f"--band-deltas output/band_deltas.pt --band-module {module} --band-token-mode all --steer-positions {positions} --band-alpha {alpha}"
            + (f" --band-random-seed {seed}" if seed is not None else ""))


def fmt_alpha(a):
    return f"{float(a):g}"


def plan(key, selection):
    """(cells {name: {"args", "role", "reference", ...}}, jobs {job: [(cell, scenarios)]})."""
    real, rand = selection[key]["selected"]["real"], selection[key]["selected"]["random"]
    ra, qa = fmt_alpha(real["alpha"]), fmt_alpha(rand["alpha"])
    cells = {"base": {"args": "", "role": "base"}, "adapter-seed0": {"args": "--adapter adapters/seed0", "role": "reference"}}
    if key == "kimi-dev-72b":
        cells["below-train"] = {"args": "--adapter adapters/seed0 --adapter-layers below:56 --adapter-modules q_proj,v_proj", "role": "subset", "reference": "adapter-seed0"}
        cells["hi-half"] = {"args": "--adapter adapters/seed0 --adapter-layers range:28-55 --adapter-modules q_proj,v_proj", "role": "subset", "reference": "adapter-seed0"}
        rname = f"jv-{real['band']}-a{ra}"
        cells[rname] = {"args": offset_args(key, ra, band=real["band"]), "role": "selected real", "reference": KIMI_REF[real["band"]], "dev": real}
        qname = f"rand-{rand['band']}-a{qa}-s{rand['seed']}"
        cells[qname] = {"args": offset_args(key, qa, seed=rand["seed"], band=rand["band"]), "role": "selected random",
                        "reference": KIMI_REF[rand["band"]], "dev": rand}
        mname = f"jv-{rand['band']}-a{qa}"  # the real offset at the selected random draw's band and dose
        if mname not in cells:
            cells[mname] = {"args": offset_args(key, qa, band=rand["band"]), "role": "real, matched to the selected random cell",
                            "reference": KIMI_REF[rand["band"]]}
        names = list(cells)
        s = "main,perspectives"
        jobs = {"A": [(n, s) for n in names[0:2]], "B": [(n, s) for n in names[2:4]], "C": [(n, s) for n in names[4:5] + names[6:]],
                "D": [(n, s) for n in names[5:6]]}
        return cells, jobs
    spec = STEER_MODELS[key]
    a, b = spec["band"]
    p = "qkv" if key == "qwen38-27b" else "v"
    off_ref = "adapter-seed0" if key == "qwen38-27b" else f"band-{p}"
    if key == "qwen38-27b":
        cells["below-train"] = {"args": "--adapter adapters/seed0 --adapter-layers below:32 --adapter-modules all", "role": "subset", "reference": "adapter-seed0"}
        cells["band-qkv"] = {"args": f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules in_proj_qkv", "role": "band", "reference": "adapter-seed0"}
    else:
        cells["below-train"] = {"args": f"--adapter adapters/seed0 --adapter-layers below:{spec['layer']}", "role": "subset", "reference": "adapter-seed0"}
        cells["band-v"] = {"args": f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules v_proj", "role": "band", "reference": "adapter-seed0"}
        const = "--steer-constants output/constants.pt --steer-layer $LAYER --steer-mode replace --steer-constant adapter_seed0"
        cells["replace-adapter-s0"] = {"args": f"{const} --steer-positions from_last", "role": "constant", "reference": "adapter-seed0"}
        cells["replace-adapter-s0-all"] = {"args": f"{const} --steer-positions all", "role": "constant", "reference": "adapter-seed0"}
    rname, qname = f"{p}-all-a{ra}", f"rand-all-a{qa}-s{rand['seed']}"
    cells[rname] = {"args": offset_args(key, ra), "role": "selected real", "reference": off_ref, "dev": real}
    cells[qname] = {"args": offset_args(key, qa, seed=rand["seed"]), "role": "selected random", "reference": off_ref, "dev": rand}
    pos_alphas = [ra] + (["2"] if key == "qwen38-27b" and ra != "2" else [])
    for pa in pos_alphas:
        for pos in ("prompt", "content"):
            cells[f"{p}-{pos}-a{pa}"] = {"args": offset_args(key, pa, positions=pos), "role": f"{pos} only", "reference": off_ref}
    core = ["base", "adapter-seed0", "below-train", f"band-{p}", rname, qname]
    main_job = [(n, MAIN3) for n in cells]
    var = ",".join(VARIANTS)
    if key == "mistral-7b":
        jobs = {"A": main_job, "B": [(n, var) for n in core[:3]], "C": [(n, var) for n in core[3:]]}
    else:
        jobs = {"A": main_job, "B": [(n, var) for n in core]}
    return cells, jobs


RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=soo2-heldout-__KEY__
#SBATCH --time=__HOURS__:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=__CPUS__
__GPU_HEADER__
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root: run.sh SNAPSHOT JOB (HELDOUT_ROUND.md).
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
JOB=${2:?pass the job letter}
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit through Slurm from the repository root}
export PYTHONPATH="$SNAPSHOT_ROOT/src"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
__CHAT_KWARGS__
export OMP_NUM_THREADS=__CPUS__
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not assign a visible GPU; refusing to select an unallocated GPU."
    exit 1
fi
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; job $JOB"
# On the cs nodes a task can see GPUs it was not allocated (a sibling task's); keep the least-used allocated count.
N_ALLOC=${SLURM_GPUS_ON_NODE:-1}
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt "$N_ALLOC" ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -"$N_ALLOC" | cut -d, -f1 | paste -sd,)"
    echo "more GPUs visible than allocated; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
cd "$SNAPSHOT_ROOT"
PY="$REPO_ROOT/__VENV__/bin/python"
MODEL=__MODEL__
LAYER=__LAYER__
SUFFIX=__SUFFIX__
mkdir -p output/eval
run() {  # run NAME ORIENT SCENARIOS(comma-separated) ARGS...
    local name=$1 orient=$2 scen=$3; shift 3
    local data=data/eval_heldout; [[ "$orient" == mirrored ]] && data=data/eval_heldout_mirrored
    local tag="${name}_${orient}"
    local s todo=()
    for s in ${scen//,/ }; do
        [[ -f "output/eval/${tag}_${s}_${SUFFIX}_summary.json" ]] || todo+=("$s")
    done
    if [[ ${#todo[@]} == 0 ]]; then echo "skip $tag $scen (complete)"; return; fi
    echo "=== $(date -u +%FT%TZ) start $tag ${todo[*]}"
    "$PY" -u -m selfconcept.soo.evaluate --model "$MODEL" --data "$data" --out output/eval --tag "$tag" \\
        --scenarios "${todo[@]}" --n 250 --suffix "$SUFFIX" __EVAL__ "$@"
    echo "=== $(date -u +%FT%TZ) done $tag ${todo[*]}"
}
case "$JOB" in
__CASES__
*) echo "unknown job $JOB"; exit 1 ;;
esac
echo "=== job $JOB complete"
"""


def run_sh(key, spec, cells, jobs):
    kimi = key == "kimi-dev-72b"
    gpu = ("#SBATCH --partition=cs3\n#SBATCH --gpus-per-node=1\n#SBATCH --qos=cs" if kimi
           else "#SBATCH --partition=cs2\n#SBATCH --gpus-per-node=h100:1\n#SBATCH --qos=cs")
    cases = "\n".join(f"{j})\n" + "\n".join(f'    for orient in orig mirrored; do run {n} "$orient" "{scen}" {cells[n]["args"]}; done'.replace(" ; done", "; done")
                                            for n, scen in runs) + "\n    ;;" for j, runs in jobs.items())
    return (RUN_SH.replace("__KEY__", key).replace("__HOURS__", f"{HOURS[key]:02d}").replace("__CPUS__", "8" if kimi else "4")
            .replace("__GPU_HEADER__", gpu).replace("__CHAT_KWARGS__", "" if kimi else "export SOO_CHAT_KWARGS='{\"enable_thinking\": false}'")
            .replace("__VENV__", VENV.name).replace("__MODEL__", spec["model"]).replace("__LAYER__", str(spec["layer"]))
            .replace("__SUFFIX__", spec["suffix"]).replace("__EVAL__", KIMI_EVAL if kimi else "").replace("__CASES__", cases))


def check_heldout_data(scenarios):
    info = {}
    for d in ("data/eval_heldout", "data/eval_heldout_mirrored"):
        for s in scenarios:
            f = SOO / d / f"{s}.jsonl"
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if len(rows) != 250 or not all("honest_answer" in r and "deceptive_answer" in r for r in rows):
                raise SystemExit(f"{f}: need 250 rows with reference answers")
            info[f"{d}/{s}"] = {"rows": len(rows), "sha256": sha256(f)}
    return info


def freeze(snapshot, key, spec, cells, jobs, selection_dir):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    shutil.copy2(SOO / TRAIN_FILE, snapshot / "data/train_soo_pairs.jsonl")
    for d in ("eval_heldout", "eval_heldout_mirrored"):
        shutil.copytree(SOO / "data" / d, snapshot / "data" / d)
    shutil.copytree(SOO / spec["adapter_dir"] / "seed0", snapshot / "adapters" / "seed0")
    for name in ("HELDOUT_ROUND.md", "scripts/analyze_layer_round.py", "scripts/analyze_kimi_loc.py", "scripts/intent_language_check.py",
                 "scripts/select_heldout.py", "scripts/make_constants.py", "scripts/extract_band_deltas.py"):  # the last two for the tests
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_heldout_round.py")
    (snapshot / "output/eval").mkdir(parents=True)
    for ext in ("pt", "json"):
        src = STUDY2 / DELTAS[key] / f"output/band_deltas.{ext}"
        if src.exists():
            shutil.copy2(src, snapshot / f"output/band_deltas.{ext}")
    if key in CONSTANTS:
        shutil.copy2(STUDY2 / CONSTANTS[key] / "output/constants.pt", snapshot / "output/constants.pt")
    shutil.copytree(selection_dir, snapshot / "selection")
    (snapshot / "run.sh").write_text(run_sh(key, spec, cells, jobs))
    (snapshot / "run.sh").chmod(0o755)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted([*STEER_MODELS, "kimi-dev-72b"]))
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--partition"); ap.add_argument("--qos"); ap.add_argument("--gres", help="e.g. h100:1 (replaces --gpus-per-node)")
    args = ap.parse_args()
    key = args.model_key
    spec = KIMI_SPEC if key == "kimi-dev-72b" else STEER_MODELS[key]
    selection = json.loads((args.selection / "selection.json").read_text())
    cells, jobs = plan(key, selection)
    scenarios = ["main", "perspectives"] if key == "kimi-dev-72b" else MAIN3.split(",") + VARIANTS
    eval_data = check_heldout_data(scenarios)
    try:
        adapter = check_adapter(spec)
    except SystemExit as e:  # the Kimi sweep's adapters were trained by its own launcher; record why the generic check failed
        if key != "kimi-dev-72b":
            raise
        d = SOO / spec["adapter_dir"] / "seed0"
        if not (d / "adapter_config.json").exists() or not (d / "adapter_model.safetensors").exists():
            raise SystemExit(f"incomplete adapter directory: {d}")
        adapter = {"path": str(d.relative_to(ROOT)), "generic_check": str(e), "adapter_config": json.loads((d / "adapter_config.json").read_text())}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = STUDY2 / f"heldout-{key}-L{spec['layer']}-{stamp}"
    freeze(snapshot, key, spec, cells, jobs, args.selection)
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission", "snapshot": str(snapshot),
        "code_commit": commit, "working_tree": dirty, "plan": "HELDOUT_ROUND.md", "round": "heldout", "python": str(VENV / "bin/python"),
        "model_key": key, **{k: v for k, v in spec.items() if k != "band"}, "band": list(spec.get("band", ())),
        "selection": str(args.selection), "cells": cells, "jobs": {j: [list(r) for r in runs] for j, runs in jobs.items()},
        "band_deltas_from": DELTAS[key], "constants_from": CONSTANTS.get(key), "eval_args": KIMI_EVAL if key == "kimi-dev-72b" else "",
        "orientations": ["orig", "mirrored"], "scenarios": scenarios, "n_per_scenario": 250, "adapter": adapter, "eval_data": eval_data,
        "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    for j, runs in jobs.items():
        print(f"  job {j}: " + ", ".join(f"{n} [{s if s in (MAIN3, 'main,perspectives') else 'variants'}]" for n, s in runs))
    if not args.no_submit:
        overrides = [*(["--partition", args.partition] if args.partition else []), *(["--qos", args.qos] if args.qos else []),
                     *([f"--gpus-per-node={args.gres}"] if args.gres else [])]
        subs = []
        for j in jobs:
            proc = subprocess.run(["sbatch", "--parsable", *overrides, str(snapshot / "run.sh"), str(snapshot), j],
                                  cwd=ROOT, text=True, capture_output=True, check=True)
            job = proc.stdout.strip().split(";")[0]
            subs.append({"job_id": job, "job": j, "log": str(SOO / "slurm-logs" / f"soo2-heldout-{key}-{job}.out")})
            print(f"submitted job {job} ({j})")
        (snapshot / "submission.json").write_text(json.dumps({"jobs": subs, "sbatch_overrides": overrides,
                                                              "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
