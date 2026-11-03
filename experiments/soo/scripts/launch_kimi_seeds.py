"""Freeze and submit the Kimi-Dev-72B part of the control round (CONTROL_ROUND.md): random draws 1-4 of the joint q+v
offsets at the two cells where the seed-0 draw matched or beat the real offset (rand-all-a0.75 over 0-55, rand-hi-a0.5
over 28-55).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_kimi_seeds.py [--no-submit] [--partition P --qos Q --gres G]
One frozen snapshot, four jobs (one per seed). Base, the adapter, the LoRA subsets, the real offsets and the seed-0 random
cells are carried over from the joint-offset snapshot, whose band_deltas.pt is copied in. Every cell is main +
perspectives in both orientations (n=250, 1024 new tokens, classification after the think block). Readout:
analyze_kimi_joint.py (verdicts, think rates) then analyze_kimi_seeds.py (the round's decision).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from launch_bisect_round import ROOT, SOO, VENV, check_eval_data, sha256, software_checks
from launch_kimi_joint import BANDS, LAYER, RUN_SH as JOINT_RUN_SH, SCENARIOS, SPEC, cell, check_deltas
from launch_kimi_round import CONFIG, EVAL, check_model

PRIOR = "kimijoint-kimi-dev-72b-L56-20261001T202619Z"  # joint-offset round: band deltas, references, seed-0 cells
SEEDS = (1, 2, 3, 4)
CARRIED = ("base", "adapter-seed0", "below-train", "hi-half", "jv-all-a0.75", "jv-hi-a0.5", "rand-all-a0.75", "rand-hi-a0.5")
CELLS = {"jv-all-a0.75": cell("all", 0.75), "jv-hi-a0.5": cell("hi", 0.5),
         "rand-all-a0.75": {**cell("all", 0.75, random=True), "seed": 0}, "rand-hi-a0.5": {**cell("hi", 0.5, random=True), "seed": 0}}
for _s in SEEDS:
    CELLS[f"rand-all-a0.75-s{_s}"] = {**cell("all", 0.75, random=True), "seed": _s}
    CELLS[f"rand-hi-a0.5-s{_s}"] = {**cell("hi", 0.5, random=True), "seed": _s}
for _name, _c in CELLS.items():
    _c["control"] = {"jv-all-a0.75": "rand-all-a0.75", "jv-hi-a0.5": "rand-hi-a0.5"}.get(_name)
JOBS = {str(s): [f"rand-all-a0.75-s{s}", f"rand-hi-a0.5-s{s}"] for s in SEEDS}
GROUPS = {"all-a0.75": ("jv-all-a0.75", ["rand-all-a0.75"] + [f"rand-all-a0.75-s{s}" for s in SEEDS]),
          "hi-a0.5": ("jv-hi-a0.5", ["rand-hi-a0.5"] + [f"rand-hi-a0.5-s{s}" for s in SEEDS])}


def condition_args(name):
    c = CELLS[name]
    return (f"--band-deltas output/band_deltas.pt --band-token-mode all --band-module {c['module']} --band-layers {BANDS[c['band']][0]} "
            f"--steer-positions {c['positions']} --band-alpha {c['alpha']} --band-random-seed {c['seed']}")


# On the cs nodes a task can see GPUs it was not allocated (a sibling task's); keep the least-used allocated count.
PIN = """N_ALLOC=${SLURM_GPUS_ON_NODE:-1}
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt "$N_ALLOC" ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -"$N_ALLOC" | cut -d, -f1 | paste -sd,)"
    echo "more GPUs visible than allocated; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
"""


def run_sh(hours):
    head, _, _ = JOINT_RUN_SH.partition("case \"$JOB\" in")
    head = (head.replace("#SBATCH --job-name=soo2-kimi-joint", "#SBATCH --job-name=soo2-ctrl-kimi")
            .replace("# Submit from the repository root: run.sh SNAPSHOT A|B|C.", "# Submit from the repository root: run.sh SNAPSHOT SEED.")
            .replace("(A, B or C)", "(the random seed, 1-4)")
            .replace('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv\n', PIN)
            # qos cs on one cs3 B200 (180 GB holds the 145 GB bf16 model); fairshare 0.1 against dw87's 1.0
            .replace("#SBATCH --gpus-per-node=a100:4\n#SBATCH --qos=dw87\n#SBATCH --exclude=dw-2-4\n",
                     "#SBATCH --partition=cs3\n#SBATCH --gpus-per-node=1\n#SBATCH --qos=cs\n"))
    assert "dw87" not in head
    cases = "\n".join(f"{j})\n" + "\n".join(f'    for orient in orig mirrored; do run {n} "$orient" "{SCENARIOS}" {condition_args(n)}; done'
                                            for n in names) + "\n    ;;" for j, names in JOBS.items())
    tail = ('*) echo "unknown job $JOB"; exit 1 ;;\nesac\n'
            '"$PY" analyze_kimi_joint.py output launch.json | tee "output/analysis_$JOB.log"\n'
            '"$PY" analyze_kimi_seeds.py output launch.json | tee "output/seeds_analysis_$JOB.log"\n'
            'echo "=== job $JOB complete"\n')
    script = head + 'case "$JOB" in\n' + cases + "\n" + tail
    return (script.replace("__VENV__", VENV.name).replace("__HOURS__", f"{hours:02d}").replace("__MODEL__", SPEC["model"])
            .replace("__EVAL__", EVAL))


def freeze(snapshot, hours, prior):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    for d in ("eval", "eval_mirrored"):
        (snapshot / "data" / d).mkdir()
        for s in SCENARIOS.split(","):
            shutil.copy2(SOO / "data" / d / f"{s}.jsonl", snapshot / "data" / d / f"{s}.jsonl")
    for name in ("CONTROL_ROUND.md", CONFIG, "scripts/analyze_layer_round.py", "scripts/analyze_kimi_loc.py", "scripts/analyze_kimi_joint.py",
                 "scripts/analyze_kimi_seeds.py", "scripts/make_constants.py", "scripts/extract_band_deltas.py"):  # the last two for the tests
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_kimi_seeds.py")
    (snapshot / "output/eval").mkdir(parents=True)
    for ext in ("pt", "json"):
        shutil.copy2(prior / f"output/band_deltas.{ext}", snapshot / f"output/band_deltas.{ext}")
    (snapshot / "run.sh").write_text(run_sh(hours))
    (snapshot / "run.sh").chmod(0o755)


def carry_over(snapshot, prior):
    out, names = snapshot / "output/eval", []
    for cond in CARRIED:
        for orient in ("orig", "mirrored"):
            for scen in SCENARIOS.split(","):
                for tail in ("_summary.json", ".jsonl"):
                    src = prior / "output/eval" / f"{cond}_{orient}_{scen}_{SPEC['suffix']}{tail}"
                    if not src.exists():
                        raise SystemExit(f"missing {src}")
                    shutil.copy2(src, out / src.name)
                    names.append(src.name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--partition"); ap.add_argument("--qos"); ap.add_argument("--gres", help="e.g. 1 or h100:3 (replaces --gpus-per-node)")
    args = ap.parse_args()
    model_info = check_model(SPEC)
    results = (SOO / "results/study2").resolve()
    prior = results / PRIOR
    deltas = check_deltas(prior / "output/band_deltas.pt")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = results / f"ctrl-kimi-dev-72b-L{LAYER}-{stamp}"
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
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "CONTROL_ROUND.md", "round": "control-kimi",
        "python": str(VENV / "bin/python"), "model_key": "kimi-dev-72b", **SPEC,
        "conditions": conditions, "cells": CELLS, "jobs": JOBS, "groups": GROUPS, "carried_from": PRIOR, "carried_over": carried,
        "band_deltas": deltas, "model_check": model_info, "eval_args": EVAL,
        "orientations": ["orig", "mirrored"], "scenarios": SCENARIOS.split(","), "n_per_scenario": 250,
        "eval_data": eval_data, "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        overrides = [*(["--partition", args.partition] if args.partition else []), *(["--qos", args.qos] if args.qos else []),
                     *([f"--gpus-per-node={args.gres}"] if args.gres else [])]
        jobs = []
        for letter in JOBS:
            proc = subprocess.run(["sbatch", "--parsable", *overrides, str(snapshot / "run.sh"), str(snapshot), letter],
                                  cwd=ROOT, text=True, capture_output=True, check=True)
            job = proc.stdout.strip().split(";")[0]
            jobs.append({"job_id": job, "job": letter, "conditions": JOBS[letter], "log": str(SOO / "slurm-logs" / f"soo2-ctrl-kimi-{job}.out")})
            print(f"submitted job {job} (seed {letter}: {', '.join(JOBS[letter])})")
        (snapshot / "submission.json").write_text(json.dumps({"jobs": jobs, "sbatch_overrides": overrides,
                                                              "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
