#!/bin/bash --login
#SBATCH --job-name=schemer-rh-smoke
#SBATCH --time=04:00:00
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

# Phase 3 smoke (PLAN.md): HF-backend throughput and backend check for the clean-protocol recipe on gpt-oss-120b.
# 1. Token-mean residual norms at every layer over the turn-0 prompts of the 45 hard problems (layer_norms.pt).
# 2. Unsteered episodes on the first N problems of ib_solvhard_s0, seed 0, recipe settings (low effort, prompt C +
#    no-carve, 5 attempts, stated budget, neutral feedback, T=1, 16384 new tokens), with the layer-17 projection
#    sidecar onto DIRECTION (any (36, 2880) axis; the Assistant Axis by default, until a schemer vector exists).
# Usage: sbatch --export=ALL[,N=3][,DIRECTION=<axis.pt>][,ROOT=<dir>] experiments/schemer-axis/slurm/steered_rh_smoke.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
nvidia-smi

MODEL="openai/gpt-oss-120b"
REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
DIRECTION="${DIRECTION:-$MIRROR/assistant-axis/gptoss12b-thinking/axis_response_only.py}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/rh_smoke}"
N="${N:-3}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed 0)

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

echo "########## Layer norms ##########"
python scripts/steered_rh.py --model "$MODEL" --revision "$REVISION" --measure-norms \
    --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 "${RECIPE[@]}" --out "$ROOT/norms"

echo "########## Unsteered episodes ($N problems) ##########"
SECONDS=0
python scripts/steered_rh.py --model "$MODEL" --revision "$REVISION" --direction "$DIRECTION" \
    --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" --projection-layers 17 --out "$ROOT/unsteered" --tag ib_solvhard_s0
echo "episodes took ${SECONDS}s"
python - "$ROOT/unsteered" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
turns = [json.loads(l) for l in (root / "ib_solvhard_s0_ib_solvhard_s0_projection.jsonl").open()]
gen = sum(t["generated_tokens"] for t in turns)
print(f"{len(turns)} turns, {gen} generated tokens, {sum(t['prompt_tokens'] for t in turns)} prompt tokens")
for r in (json.loads(l) for l in (root / "ib_solvhard_s0_impossible_original.jsonl").open()):
    print(r["example_id"], r["label"], r["n_attempts"], "attempts", "passed_original" if r.get("passed_original") else "")
PY

echo "=== schemer-rh-smoke done: $ROOT ==="
