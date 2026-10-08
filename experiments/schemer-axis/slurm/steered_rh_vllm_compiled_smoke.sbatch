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
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {d: [json.loads(l) for l in (root / d / "probe.jsonl").open()] for d in ["probe_eager", "probe_compiled", "probe_unsteered"]}
for e, c, u in zip(rows["probe_eager"], rows["probe_compiled"], rows["probe_unsteered"]):
    same = e["final"] == c["final"]
    pre = next((i for i, (a, b) in enumerate(zip(e["final"], c["final"])) if a != b), min(len(e["final"]), len(c["final"])))
    print(f"eager==compiled: {same} (common prefix {pre} chars of {len(e['final'])}/{len(c['final'])}); compiled==unsteered: {c['final'] == u['final']}  | {e['question'][:50]}")
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
