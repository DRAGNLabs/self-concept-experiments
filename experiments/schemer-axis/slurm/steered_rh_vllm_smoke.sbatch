#!/bin/bash --login
#SBATCH --job-name=schemer-vllm-smoke
#SBATCH --time=03:00:00
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

# Phase 3 backend check on vLLM (PLAN.md): the HF path decodes at ~10 tok/s, vLLM ~18x faster single-stream.
# 1. Hook check: layer-17 token-mean projection/norm of the turn-0 prompts of the first 3 hard problems through a
#    recording hook on the vLLM block, to compare with the HF projection sidecar (rh_smoke: 447/442/422, norms ~8.2k).
# 2. Unsteered: 15 episodes of ib_solvhard_s0, seed 0, recipe settings, for throughput and a parity look.
# 3. Add arm (alpha +0.4 on DIRECTION) and cap arm (tau=300 at layer 17) on the first 3 problems: hooks fire, no crash.
# 4. Probe: free-text answers under alpha -0.4 / 0 / +0.4 (steering sanity by eye).
# Usage: sbatch --export=ALL[,DIRECTION=<axis.pt>][,ROOT=<dir>][,N=15] experiments/schemer-axis/slurm/steered_rh_vllm_smoke.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
DIRECTION="${DIRECTION:-$MIRROR/assistant-axis/gptoss12b-thinking/axis_response_only.py}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/vllm_smoke}"
NORMS="${NORMS:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/rh_smoke/norms/layer_norms.pt}"
N="${N:-15}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed 0)
mkdir -p "$ROOT"
echo '{"17": 300.0}' > "$ROOT/cap_thresholds.json"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

echo "########## Hook check (3 prompts) ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --check-hook --scenarios ib_solvhard_s0 --n 3 \
    "${RECIPE[@]}" --out "$ROOT/hook_check" --tag check

echo "########## Unsteered episodes ($N problems) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" \
    --out "$ROOT/unsteered" --tag ib_solvhard_s0
echo "unsteered episodes took ${SECONDS}s"

echo "########## Add arm alpha=+0.4 (3 problems) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --layer-norms "$NORMS" \
    --steer-mode add --steer-layer 17 --steer-alpha 0.4 --scenarios ib_solvhard_s0 --n 3 "${RECIPE[@]}" \
    --out "$ROOT/add_p0.4" --tag ib_solvhard_s0
echo "add episodes took ${SECONDS}s"

echo "########## Cap arm tau=300 at layer 17 (3 problems) ##########"
python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" \
    --steer-mode cap --cap-layers 17 --cap-thresholds "$ROOT/cap_thresholds.json" --scenarios ib_solvhard_s0 --n 3 "${RECIPE[@]}" \
    --out "$ROOT/cap_300" --tag ib_solvhard_s0

echo "########## Probe ##########"
PROBES=("What is the relationship between law and morality?" "How do you approach a test you are not prepared for?" "Tell me about yourself.")
for A in -0.4 0.4; do
    python scripts/steered_rh_vllm.py --model "$MODEL" --direction "$DIRECTION" --layer-norms "$NORMS" \
        --steer-mode add --steer-layer 17 --steer-alpha "$A" --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
        --out "$ROOT/probe" --tag probe --probe "${PROBES[@]}"
done
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
    --out "$ROOT/probe" --tag probe --probe "${PROBES[@]}"

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for arm in ["unsteered", "add_p0.4", "cap_300"]:
    p = root / arm / "ib_solvhard_s0_impossible_original.jsonl"
    if not p.exists():
        print(arm, "no records"); continue
    recs = [json.loads(l) for l in p.open()]
    print(f"{arm}: {len(recs)} episodes;", {k: sum(r['label'] == k for r in recs) for k in sorted({r['label'] for r in recs})},
          "passed_original", sum(bool(r.get('passed_original')) for r in recs), "turns", sum(r['n_attempts'] for r in recs))
PY
echo "=== schemer-vllm-smoke done: $ROOT ==="
