"""Freeze and submit a band-steering round for one model (STEER_ROUND.md, DOSE_ROUND.md, CONTROL_ROUND.md).

Usage, from the repository root:
  .venv-313/bin/python experiments/soo/scripts/launch_steer_round.py --model-key gemma4-31b [--round steer|dose|control] [--no-submit]
      [--partition P --qos Q --gres GRES]  (sbatch overrides of run.sh's dw87 / a100 header)
--round steer (default) is the band-steering round (Qwen3.8-27B: its addendum, steering the fused DeltaNet
in_proj_qkv); --round dose is the dose/position round (alpha between 1 and 2, content-token and prompt-only
offsets; no gap measurement); --round control is the control round (random seeds 1-4 at the doses the claims
rest on, and from_last offsets with the all-token mean; no gap measurement). The job measures the band's mean LoRA deltas on the training prompts (extract_band_deltas.py), evaluates the
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
    # Band from the second bisection (BISECT2_ROUND.md): 20-31 is the smallest reproducing union; 22-31 and 16-29 are not.
    "gemma4-31b": dict(model="google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", layer=32,
                       adapter_dir="results/checkpoints/gemma4-31b-L32", suffix="room_only", hours=16,
                       band=(20, 31), bisect="u20 (20-31): main D 0/2, TH D 100/100 (adapter 0/6), perspectives 100/100, no refusals"),
    "gemma4-12b": dict(model="google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", layer=24,
                       adapter_dir="results/checkpoints/gemma4-12b-L24", suffix="room_only", hours=8,
                       band=(19, 23), bisect="band (19-23): main D 0/0, TH 0/0, perspectives 100/100, no refusals"),
    "mistral-7b": dict(model="mistralai/Mistral-7B-Instruct-v0.2", revision="63a8b081895390a26e140280378bc85ec8bce07a", layer=16,
                       adapter_dir="results/checkpoints/mistral-7b-lasttok-L16", suffix="i_would", hours=30,
                       band=(5, 13), bisect="band (5-13): main D 13.6/0, clean honest 75/91%"),
    # Band from the Qwen localization round (QWEN_LOC_ROUND.md): 24-31 @ all modules reproduces; the DeltaNet
    # in_proj_qkv deltas below L32 reproduce, their value slice alone is partial, the full-attention modules are null.
    "qwen38-27b": dict(model="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", layer=32,
                       adapter_dir="results/checkpoints/qwen38-27b-L32", suffix="room_only", hours=16,
                       modules=["in_proj_qkv", "q_proj", "v_proj"], extract_modules="in_proj_qkv,q_proj,v_proj",
                       reference="band-qkv", band=(24, 31),
                       bisect="q4 (24-31 @ all): main D 0/0, TH D 7/6, perspectives 100/100; below-qkv reproduces, below-qkv-v 42/56"),
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


def conditions_qwen(spec):
    """Qwen3.8-27B: the band's fused in_proj_qkv mean deltas (whole, value slice, query-key slice) at 24-31."""
    a, b = spec["band"]
    lora = f"--adapter adapters/seed0 --adapter-layers range:{a}-{b}"
    mode = lambda module, tok, pos: f"{BAND} --band-module {module} --band-token-mode {tok} --steer-positions {pos}"  # noqa: E731
    qkv_all, v_all = mode("in_proj_qkv", "all", "all"), mode("in_proj_qkv.v", "all", "all")
    return [("base", ""), ("adapter-seed0", "--adapter adapters/seed0"),
            ("band-qkv", f"{lora} --adapter-modules in_proj_qkv"), ("band-qkv-v", f"{lora} --adapter-modules in_proj_qkv.v"),
            ("qkv-resp-a1", f"{mode('in_proj_qkv', 'last', 'from_last')} --band-alpha 1"),
            ("qkv-all-a1", f"{qkv_all} --band-alpha 1"), ("qkv-all-a1.5", f"{qkv_all} --band-alpha 1.5"),
            ("qkv-all-a2", f"{qkv_all} --band-alpha 2"),
            ("v-all-a1", f"{v_all} --band-alpha 1"), ("v-all-a2", f"{v_all} --band-alpha 2"),
            ("qk-all-a1", f"{mode('in_proj_qkv.qk', 'all', 'all')} --band-alpha 1"),
            ("rand-all-a1", f"{qkv_all} --band-alpha 1 --band-random-seed 0"), ("rand-all-a2", f"{qkv_all} --band-alpha 2 --band-random-seed 0"),
            ("qkv-content-a1", f"{mode('in_proj_qkv', 'all', 'content')} --band-alpha 1"),
            ("qkv-prompt-a1", f"{mode('in_proj_qkv', 'all', 'prompt')} --band-alpha 1")]


def conditions_dose(spec):
    """Dose/position round (DOSE_ROUND.md): alpha 1.25-1.75 at every position with matched randoms, and the
    'all' mean restricted to the user message's tokens or to the prompt."""
    a, b = spec["band"]
    v = lambda pos: f"{BAND} --band-module v_proj --band-token-mode all --steer-positions {pos}"  # noqa: E731
    out = [("base", ""), ("band-v", f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules v_proj"),
           ("v-all-a1", f"{v('all')} --band-alpha 1")]
    for alpha in ("1.25", "1.5", "1.75"):
        out.append((f"v-all-a{alpha}", f"{v('all')} --band-alpha {alpha}"))
    for alpha in ("1.25", "1.5", "1.75"):
        out.append((f"rand-all-a{alpha}", f"{v('all')} --band-alpha {alpha} --band-random-seed 0"))
    return out + [("v-prompt-a1", f"{v('prompt')} --band-alpha 1"), ("v-content-a1", f"{v('content')} --band-alpha 1"),
                  ("v-content-a2", f"{v('content')} --band-alpha 2"), ("rand-content-a1", f"{v('content')} --band-alpha 1 --band-random-seed 0")]


def overlap_runs_qwen(spec):
    a, b = spec["band"]
    band = (f"{BAND} --band-modules in_proj_qkv in_proj_qkv.v in_proj_qkv.qk --band-random-seeds 0 "
            f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules in_proj_qkv")
    return [("resp-a1", f"{band} --positions from_last --band-token-mode last --band-alpha 1"),
            ("all-a1", f"{band} --positions all --band-token-mode all --band-alpha 1"),
            ("all-a2", f"{band} --positions all --band-token-mode all --band-alpha 2"),
            ("full-adapter", "--adapter adapters/seed0")]


def conditions_control(key, spec):
    """Control round (CONTROL_ROUND.md): random draws 1-4 at the doses the paper's direction claims rest on, and the
    all-token mean applied from the last prompt token onward (same vector as v-all, different positions)."""
    a, b = spec["band"]
    if key == "qwen38-27b":
        mode = lambda pos: f"{BAND} --band-module in_proj_qkv --band-token-mode all --steer-positions {pos}"  # noqa: E731
        out = [("base", ""), ("adapter-seed0", "--adapter adapters/seed0"),
               ("band-qkv", f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules in_proj_qkv"),
               ("qkv-all-a2", f"{mode('all')} --band-alpha 2")]
        real = "qkv"
    else:
        mode = lambda pos: f"{BAND} --band-module v_proj --band-token-mode all --steer-positions {pos}"  # noqa: E731
        out = [("base", ""), ("band-v", f"--adapter adapters/seed0 --adapter-layers range:{a}-{b} --adapter-modules v_proj"),
               ("v-all-a1", f"{mode('all')} --band-alpha 1")]
        real = "v"
    for alpha in CONTROL_ALPHAS[key]:
        for seed in CONTROL_SEEDS.get((key, alpha), RANDOM_SEEDS):
            out.append((f"rand-all-a{alpha}-s{seed}", f"{mode('all')} --band-alpha {alpha} --band-random-seed {seed}"))
    out += [(f"{real}-last-a1", f"{mode('from_last')} --band-alpha 1"), (f"{real}-last-a2", f"{mode('from_last')} --band-alpha 2")]
    if key == "qwen38-27b":
        out += [("qkv-prompt-a2", f"{mode('prompt')} --band-alpha 2"), ("qkv-content-a2", f"{mode('content')} --band-alpha 2")]
    return out


CONTROL_ALPHAS = {"mistral-7b": ("1", "2"), "gemma4-12b": ("1", "1.25"), "gemma4-31b": ("1", "1.5", "2"), "qwen38-27b": ("1.5", "2")}
RANDOM_SEEDS = (1, 2, 3, 4)
CONTROL_SEEDS = {("qwen38-27b", "1.5"): (0, 1, 2, 3, 4)}  # Qwen's steering round had no seed-0 random cell at alpha 1.5
# Earlier snapshots holding each model's seed-0 random cells and real all-position offsets at the control doses; the
# analysis reads each against its own snapshot's same-day base and reference.
CONTROL_PRIOR = {
    "mistral-7b": ["steer-mistral-7b-L16-20260928T230233Z"],
    "gemma4-12b": ["steer-gemma4-12b-L24-20260928T230232Z", "dose-gemma4-12b-L24-20260929T221158Z"],
    "gemma4-31b": ["steer-gemma4-31b-L32-20260929T180338Z", "dose-gemma4-31b-L32-20260929T221420Z"],
    "qwen38-27b": ["steer-qwen38-27b-L32-20260929T221532Z"],
}
# Wall-clock limits for the control round (the qos cs limit is 24 h; the dose round's Mistral job took 6.2 h for 12 cells).
CONTROL_HOURS = {"mistral-7b": 14, "gemma4-12b": 6, "gemma4-31b": 8, "qwen38-27b": 8}

ROUNDS = {"steer": dict(plan="STEER_ROUND.md", prefix="steer"), "dose": dict(plan="DOSE_ROUND.md", prefix="dose"),
          "control": dict(plan="CONTROL_ROUND.md", prefix="ctrl")}


def round_conditions(round_name, key, spec):
    if round_name == "control":
        return conditions_control(key, spec), []
    if round_name == "dose":
        if key == "qwen38-27b":
            raise SystemExit("the dose round covers gemma4-12b, mistral-7b and gemma4-31b; Qwen's dose cells are in its steering round")
        return conditions_dose(spec), []
    if key == "qwen38-27b":
        return conditions_qwen(spec), overlap_runs_qwen(spec)
    return conditions(spec), overlap_runs(spec)


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
# On the cs nodes a one-GPU task can see a sibling task's GPU; pin the least-used visible GPU by UUID.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
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
        --modules __EXTRACT_MODULES__ --pairs data/train_soo_pairs.jsonl --out output/band_deltas.pt
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
__ANALYZE__
echo "=== all runs complete"
"""


def freeze(snapshot, key, spec, n_layers, round_name="steer"):
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
    for name in (ROUNDS[round_name]["plan"], "scripts/make_constants.py", "scripts/extract_band_deltas.py", "scripts/analyze_layer_round.py",
                 "scripts/analyze_steer_round.py", "scripts/analyze_control_round.py", "scripts/intent_language_check.py"):
        shutil.copy2(SOO / name, snapshot / Path(name).name)
    shutil.copy2(Path(__file__), snapshot / "launch_steer_round.py")
    a, b = spec["band"]
    conds, overlap_list = round_conditions(round_name, key, spec)
    runs = "\n".join(f'    run {name} "$orient" "$ALL" {args}'.rstrip() for name, args in conds)
    overlaps = "\n".join(f"overlap {name} {args}" for name, args in overlap_list)
    hours = CONTROL_HOURS[key] if round_name == "control" else spec["hours"]
    analyze = '"$PY" analyze_steer_round.py output launch.json | tee output/analysis.log'
    if round_name == "control":
        analyze += '\n"$PY" analyze_control_round.py output launch.json | tee output/control_analysis.log'
    script = (RUN_SH.replace("__VENV__", VENV.name).replace("__JOB__", f"soo2-{ROUNDS[round_name]['prefix']}-{key}").replace("__HOURS__", f"{hours:02d}")
              .replace("__MODEL__", spec["model"]).replace("__REVISION__", spec["revision"]).replace("__LAYER__", str(spec["layer"]))
              .replace("__SUFFIX__", spec["suffix"]).replace("__BAND_SPEC__", f"range:{a}-{b}").replace("__BAND_TOP__", str(b))
              .replace("__LAST_LAYER__", str(n_layers - 1)).replace("__PROBES__", PROBES)
              .replace("__RUNS__", runs).replace("__OVERLAPS__", overlaps).replace("__ANALYZE__", analyze)
              .replace("__EXTRACT_MODULES__", spec.get("extract_modules", "q_proj,v_proj")))
    if round_name == "control":  # qos cs on the cs2 H100s (fairshare 0.1 against dw87's 1.0); see CONTROL_ROUND.md
        header = "#SBATCH --gpus-per-node=a100:1\n#SBATCH --qos=dw87\n#SBATCH --exclude=dw-2-4\n"
        assert script.count(header) == 1
        script = script.replace(header, "#SBATCH --partition=cs2\n#SBATCH --gpus-per-node=h100:1\n#SBATCH --qos=cs\n")
    (snapshot / "run.sh").write_text(script)
    (snapshot / "run.sh").chmod(0o755)


def submit(snapshot, key, prefix="steer", overrides=()):
    proc = subprocess.run(["sbatch", "--parsable", *overrides, str(snapshot / "run.sh"), str(snapshot)], cwd=ROOT, text=True, capture_output=True, check=True)
    job = proc.stdout.strip().split(";")[0]
    (snapshot / "submission.json").write_text(json.dumps({"job_id": job, "submitted_utc": datetime.now(timezone.utc).isoformat(), "sbatch_overrides": list(overrides),
                                                         "log": str(SOO / "slurm-logs" / f"soo2-{prefix}-{key}-{job}.out")}, indent=2) + "\n")
    print(f"submitted job {job}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--round", choices=sorted(ROUNDS), default="steer")
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--partition"); ap.add_argument("--qos"); ap.add_argument("--gres", help="e.g. gpu:h100:1 (replaces --gpus-per-node)")
    args = ap.parse_args()
    key, spec = args.model_key, MODELS[args.model_key]
    prefix = ROUNDS[args.round]["prefix"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = (SOO / "results/study2").resolve() / f"{prefix}-{key}-L{spec['layer']}-{stamp}"  # real path (results may be a symlink)
    adapter = check_adapter(spec)
    eval_data = check_eval_data()
    if spec["band"][1] >= spec["layer"]:
        raise SystemExit("the band must lie below the training layer")
    freeze(snapshot, key, spec, adapter["n_layers"], args.round)
    test_result = software_checks(snapshot)
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout
    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_submission",
        "snapshot": str(snapshot), "code_commit": commit, "working_tree": dirty, "plan": ROUNDS[args.round]["plan"], "round": args.round,
        "python": str(VENV / "bin/python"), "model_key": key, **{k: v for k, v in spec.items() if k != "band"},
        "band_layers": list(range(spec["band"][0], spec["band"][1] + 1)), "n_layers": adapter["n_layers"],
        "conditions": dict(round_conditions(args.round, key, spec)[0]), "overlap_runs": dict(round_conditions(args.round, key, spec)[1]),
        "reference": spec.get("reference", "band-v"),
        **({"control_alphas": list(CONTROL_ALPHAS[key]), "control_prior": CONTROL_PRIOR[key],
            "real_prefix": "qkv" if key == "qwen38-27b" else "v"} if args.round == "control" else {}),
        "orientations": ["orig", "mirrored"], "scenarios": ["main", "treasure_hunt", "perspectives"], "n_per_scenario": 250,
        "probes": PROBES, "adapter": adapter, "eval_data": eval_data, "test_result": test_result,
        "files_sha256": {str(p.relative_to(snapshot)): sha256(p) for p in files},
    }
    (snapshot / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"frozen {snapshot}\n{test_result}")
    if not args.no_submit:
        overrides = [*(["--partition", args.partition] if args.partition else []), *(["--qos", args.qos] if args.qos else []),
                     *([f"--gpus-per-node={args.gres}"] if args.gres else [])]
        submit(snapshot, key, prefix, overrides)


if __name__ == "__main__":
    main()
