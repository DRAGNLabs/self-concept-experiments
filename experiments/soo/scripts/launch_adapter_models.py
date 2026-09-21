"""Freeze and submit the existing-adapter measurement for another model (ADAPTER_ROUND.md, round 2c).

Usage, from the repository root:
  .venv/bin/python experiments/soo/scripts/launch_adapter_models.py --model-key gemma4-31b [--no-submit]
Each job measures base, the model's additive vector with three random controls, and every seed of the
model's validated LoRA cell, at the final prompt token and at the last user-content token.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOO = ROOT / "experiments/soo"
TRAIN_FILE = "data/train_soo_pairs.jsonl"

MODELS = {
    "gemma4-31b": dict(model="google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", layer=32,
                       residual=[32, 39, 46, 53, 59], adapter_dir="results/checkpoints/gemma4-31b-L32", seeds=(0, 1, 2),
                       vectors="results/steering/gemma4_31b.pt", alpha=16, first_study="room task 100/100 both orientations, n=250 seed 0; L30 steering α16 validated"),
    "gemma4-12b": dict(model="google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", layer=24,
                       residual=[24, 30, 36, 42, 47], adapter_dir="results/checkpoints/gemma4-12b-L24", seeds=(0, 1, 2),
                       vectors="results/steering/gemma4_12b.pt", alpha=12, first_study="room task main 74–98 orig / 58–90 mirrored, three seeds; L19 steering α12 validated"),
    "mistral-7b": dict(model="mistralai/Mistral-7B-Instruct-v0.2", revision="63a8b081895390a26e140280378bc85ec8bce07a", layer=16,
                       residual=[16, 20, 24, 28, 31], adapter_dir="results/checkpoints/mistral-7b-lasttok-L16", seeds=(0, 1, 2, 3, 4),
                       vectors="results/steering/mistral7b.pt", alpha=8, first_study="main deceptive 90.4 → 9.9 ± 6.1, five seeds n=250; steering inert at L16"),
}

RUN_SH = """#!/bin/bash --login
#SBATCH --job-name=__JOB__
#SBATCH --time=03:00:00
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
run() {  # run OFFSET NAME ARGS...
    local off=$1 name=$2; shift 2
    local out="output/off$off/$name"
    if [[ -f "$out/manifest.json" ]]; then echo "skip $out (complete)"; return; fi
    rm -rf "$out"; mkdir -p "output/off$off"
    echo "=== $(date -u +%FT%TZ) start off$off $name"
    "$PY" -u -m selfconcept.soo.measure_overlap __COMMON__ --endpoint-offset "$off" "$@" --out "$out"
    echo "=== $(date -u +%FT%TZ) done off$off $name"
}
for off in __OFFSETS__; do
    run "$off" steering-add-a__ALPHA__ --vectors data/vectors.pt --mode add --alpha __ALPHA__ --random-seeds 0 1 2
__ADAPTER_RUNS__
    "$PY" analyze_adapter_round.py "output/off$off" | tee "output/off$off/analysis.log"
done
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


def dev_pairs():
    rows = [json.loads(l) for l in (SOO / "data/subspace_pilot_pairs.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["split"] == "development"]


def check_membership():
    dev = dev_pairs()
    dev_prompts = {norm(r[k]) for r in dev for k in ("self_prompt", "other_prompt")}
    train = [json.loads(l) for l in (SOO / TRAIN_FILE).read_text().splitlines() if l.strip()]
    hits = sorted(dev_prompts & {norm(r[k]) for r in train for k in ("self_prompt", "other_prompt")})
    if hits:
        raise SystemExit(f"development prompt found in {TRAIN_FILE}: {hits[:3]}")
    return {"development_pairs": len(dev), "train_file": TRAIN_FILE, "train_pairs": len(train), "sha256": sha256(SOO / TRAIN_FILE), "literal_hits": [],
            "thematic_overlap_note": "development family 'telescope' vs training item 'professional telescope'; prompts and task differ"}


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
        info[f"seed{seed}"] = {"path": str(d.relative_to(ROOT)), "config": cfg, "seed": log["seed"],
                               "final_epoch_loss": log["epoch_losses"][-1], "n_epoch_losses": len(log["epoch_losses"])}
    return info


def check_endpoints(spec):
    """Offset 0 is the last prompt token. Find the offset that reads the last user-content token for every prompt."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("SOO_CHAT_KWARGS", '{"enable_thinking": false}')
    sys.path.insert(0, str(ROOT / "src"))
    from transformers import AutoTokenizer
    from selfconcept.common.chat import chat_template_kwargs
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    report = {}
    offsets = set()
    for r in dev_pairs():
        got = {}
        for key in ("self_prompt", "other_prompt"):
            text = tok.apply_chat_template([{"role": "user", "content": r[key]}], tokenize=False, add_generation_prompt=True, **chat_template_kwargs())
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            end = text.rfind(r[key]) + len(r[key]) - 1
            idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= end < b]
            if len(idx) != 1:
                raise SystemExit(f"cannot locate the last user-content token for {r['id']}")
            offsets.add(len(enc["input_ids"]) - 1 - idx[0])
            got[key] = (tok.decode([enc["input_ids"][idx[0]]]), tok.decode([enc["input_ids"][-1]]))
        if got["self_prompt"][0] != got["other_prompt"][0]:
            raise SystemExit(f"user-content endpoint differs within pair {r['id']}: {got}")
        for off_label, j in (("user_last", 0), ("final", 1)):
            report.setdefault(off_label, {}); report[off_label][got["self_prompt"][j]] = report[off_label].get(got["self_prompt"][j], 0) + 1
    if len(offsets) != 1:
        raise SystemExit(f"the last user-content token is not at one fixed offset: {sorted(offsets)}")
    user_last = offsets.pop()
    return {"offsets": [0, user_last], "final_token_counts": report["final"], "user_last_offset": user_last, "user_last_token_counts": report["user_last"]}


def freeze(snapshot, key, spec, offsets):
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "tests").mkdir()
    for t in (ROOT / "tests").glob("test_soo*.py"):
        shutil.copy2(t, snapshot / "tests" / t.name)
    (snapshot / "data").mkdir()
    shutil.copy2(SOO / "data/subspace_pilot_pairs.jsonl", snapshot / "data/subspace_pilot_pairs.jsonl")
    shutil.copy2(SOO / spec["vectors"], snapshot / "data/vectors.pt")
    for seed in spec["seeds"]:
        shutil.copytree(SOO / spec["adapter_dir"] / f"seed{seed}", snapshot / "adapters/original" / f"seed{seed}")
    shutil.copy2(SOO / "ADAPTER_ROUND.md", snapshot / "ADAPTER_ROUND.md")
    shutil.copy2(SOO / "scripts/analyze_adapter_round.py", snapshot / "analyze_adapter_round.py")
    shutil.copy2(Path(__file__), snapshot / "launch_adapter_models.py")
    common = (f"--model {spec['model']} --revision {spec['revision']} --probes data/subspace_pilot_pairs.jsonl --split development "
              f"--layer {spec['layer']} --residual-layers {' '.join(map(str, spec['residual']))} --positions all --token-mode last "
              "--batch-size 4 --device cuda --dtype bfloat16 --bootstrap 2000 --seed 0")
    adapter_runs = "\n".join(f'    run "$off" adapter-original-seed{seed} --adapter adapters/original/seed{seed}' for seed in spec["seeds"])
    script = (RUN_SH.replace("__JOB__", f"soo2-adapters-{key}").replace("__COMMON__", common).replace("__ALPHA__", str(spec["alpha"]))
              .replace("__OFFSETS__", " ".join(map(str, offsets))).replace("__ADAPTER_RUNS__", adapter_runs))
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
                                                         "log": str(SOO / "slurm-logs" / f"soo2-adapters-{key}-{job}.out")}, indent=2) + "\n")
    print(f"submitted job {job}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    key, spec = args.model_key, MODELS[args.model_key]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = SOO / "results/study2" / f"adapters-{key}-L{spec['layer']}-{stamp}"
    membership = check_membership()
    adapters = check_adapters(spec)
    endpoints = check_endpoints(spec)
    freeze(snapshot, key, spec, endpoints["offsets"])
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": "ADAPTER_ROUND.md (round 2c)",
        "model_key": key, **{k: v for k, v in spec.items()},
        "expected_runs": [f"off{o}/{n}" for o in endpoints["offsets"] for n in [f"steering-add-a{spec['alpha']}", *[f"adapter-original-seed{s}" for s in spec["seeds"]]]],
        "development_pairs": 64, "training_membership_check": membership, "adapters": adapters, "endpoint_check": endpoints,
        "test_result": test_result, "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"frozen {snapshot}\n{test_result}\nendpoints {endpoints['offsets']} {endpoints['user_last_token_counts']}")
    if not args.no_submit:
        submit(snapshot, key)


if __name__ == "__main__":
    main()
