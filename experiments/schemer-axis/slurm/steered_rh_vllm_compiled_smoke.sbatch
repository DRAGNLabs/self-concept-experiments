#!/bin/bash --login
#SBATCH --job-name=schemer-vllm-compiled
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Phase 3 backend check, part 2 (PLAN.md): the eager vLLM engine decodes at ~21 tok/s (smoke 14028603), so the
# intervention is now a class patch made before the engine compiles and captures the model. This job checks that
# (1) greedy answers under alpha=+0.4 agree between eager and compiled engines and differ from unsteered ones,
# (2) throughput of compiled episodes, unsteered and under the add and cap arms.
# Usage: sbatch --export=ALL[,DIRECTION=<axis.pt>][,ROOT=<dir>][,N=10] experiments/schemer-axis/slurm/steered_rh_vllm_compiled_smoke.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1  # cached graphs are keyed on the model, not on the steering patch
nvidia-smi

MODEL="openai/gpt-oss-120b"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
DIRECTION="${DIRECTION:-$MIRROR/assistant-axis/gptoss12b-thinking/axis_response_only.py}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/vllm_compiled_smoke}"
NORMS="${NORMS:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/rh_smoke/norms/layer_norms.pt}"
N="${N:-10}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --sample-seed 0)
PROBES=("What is the relationship between law and morality?" "How do you approach a test you are not prepared for?" "Tell me about yourself.")
mkdir -p "$ROOT"
echo '{"17": 300.0}' > "$ROOT/cap_thresholds.json"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

echo "########## Greedy probes: eager +0.4 ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --layer-norms "$NORMS" --eager \
    --steer-mode add --steer-layer 17 --steer-alpha 0.4 --temperature 0 --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe_eager" --tag probe --probe "${PROBES[@]}"
echo "########## Greedy probes: compiled +0.4 ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --layer-norms "$NORMS" \
    --steer-mode add --steer-layer 17 --steer-alpha 0.4 --temperature 0 --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe_compiled" --tag probe --probe "${PROBES[@]}"
echo "########## Greedy probes: compiled unsteered ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --temperature 0 --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe_unsteered" --tag probe --probe "${PROBES[@]}"
echo "########## Greedy probes: eager cap tau=300 ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --eager --temperature 0 \
    --steer-mode cap --cap-layers 17 --cap-thresholds "$ROOT/cap_thresholds.json" --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe_cap_eager" --tag probe --probe "${PROBES[@]}"
echo "########## Greedy probes: compiled cap tau=300 ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --temperature 0 \
    --steer-mode cap --cap-layers 17 --cap-thresholds "$ROOT/cap_thresholds.json" --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe_cap_compiled" --tag probe --probe "${PROBES[@]}"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
names = ["probe_eager", "probe_compiled", "probe_unsteered", "probe_cap_eager", "probe_cap_compiled"]
rows = {d: [json.loads(l) for l in (root / d / "probe.jsonl").open()] for d in names}
def prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
for i, q in enumerate(r["question"] for r in rows["probe_eager"]):
    f = {d: rows[d][i]["final"] for d in names}
    print(f"[{q[:45]}]")
    print(f"  add  eager==compiled: {f['probe_eager'] == f['probe_compiled']} (prefix {prefix(f['probe_eager'], f['probe_compiled'])} of {len(f['probe_eager'])}/{len(f['probe_compiled'])})")
    print(f"  cap  eager==compiled: {f['probe_cap_eager'] == f['probe_cap_compiled']} (prefix {prefix(f['probe_cap_eager'], f['probe_cap_compiled'])} of {len(f['probe_cap_eager'])}/{len(f['probe_cap_compiled'])})")
    print(f"  compiled add==unsteered: {f['probe_compiled'] == f['probe_unsteered']} (prefix {prefix(f['probe_compiled'], f['probe_unsteered'])});"
          f" compiled cap==unsteered: {f['probe_cap_compiled'] == f['probe_unsteered']} (prefix {prefix(f['probe_cap_compiled'], f['probe_unsteered'])})")
PY

echo "########## Compiled unsteered episodes ($N problems) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --temperature 1.0 --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" \
    --out "$ROOT/unsteered" --tag ib_solvhard_s0
echo "compiled unsteered episodes took ${SECONDS}s"

echo "########## Compiled add alpha=+0.4 ($N problems) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --layer-norms "$NORMS" --temperature 1.0 \
    --steer-mode add --steer-layer 17 --steer-alpha 0.4 --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" \
    --out "$ROOT/add_p0.4" --tag ib_solvhard_s0
echo "compiled add episodes took ${SECONDS}s"

echo "########## Compiled cap tau=300 at layer 17 (3 problems) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --temperature 1.0 \
    --steer-mode cap --cap-layers 17 --cap-thresholds "$ROOT/cap_thresholds.json" --scenarios ib_solvhard_s0 --n 3 "${RECIPE[@]}" \
    --out "$ROOT/cap_300" --tag ib_solvhard_s0
echo "compiled cap episodes took ${SECONDS}s"

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for arm in ["unsteered", "add_p0.4", "cap_300"]:
    d = root / arm
    p = d / "ib_solvhard_s0_impossible_original.jsonl"
    if not p.exists():
        print(arm, "no records"); continue
    reason = {}
    for e in map(json.loads, (d / "ib_solvhard_s0_ib_solvhard_s0_reasoning.jsonl").open()):
        reason[(e["example_id"], e["turn"])] = len(e["reasoning"])
    recs = [json.loads(l) for l in p.open()]
    chars = sum(len(a["completion"]) + reason.get((r["example_id"], t), 0) for r in recs for t, a in enumerate(r["attempts"]))
    print(f"{arm}: {len(recs)} episodes, {sum(r['n_attempts'] for r in recs)} turns, ~{chars/3.6:.0f} generated tokens;",
          {k: sum(r['label'] == k for r in recs) for k in sorted({r['label'] for r in recs})},
          "passed_original", sum(bool(r.get('passed_original')) for r in recs))
PY
echo "=== schemer-vllm-compiled done: $ROOT ==="
