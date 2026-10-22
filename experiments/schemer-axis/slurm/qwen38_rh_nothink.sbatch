#!/bin/bash --login
#SBATCH --job-name=schemer-qwen38-nothink
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=eng
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Unsteered Qwen3.8 episodes with thinking off. The 16k and 32k thinking-on smokes truncated
# mid-proof and submitted no code. The template closes </think> in the prompt, so the
# completion is the answer (split_qwen_answer). Same 3 problems.
# Usage: sbatch experiments/schemer-axis/slurm/qwen38_rh_nothink.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"enable_thinking":false}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38-27b/rh}"
N="${N:-3}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 8192 --max-model-len 32768
        --temperature 1.0 --top-p 0.95 --top-k 20 --sample-seed 0)

source ../../activate.sh

echo "########## Unsteered episodes ($N problems, thinking off) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" \
    --out "$ROOT/smoke_unsteered_nothink" --tag smoke
echo "episodes took ${SECONDS}s"
python - "$ROOT/smoke_unsteered_nothink" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
turns = [json.loads(l) for l in (root / "smoke_ib_solvhard_s0_reasoning.jsonl").open()]
print(f"{len(turns)} turns, {sum(len(t['reasoning']) for t in turns)} reasoning chars, {sum(t['truncated'] for t in turns)} truncated")
for r in (json.loads(l) for l in (root / "smoke_impossible_original.jsonl").open()):
    print(r["example_id"], r["label"], r["n_attempts"], "attempts", "passed" if r.get("passed") else "", "passed_original" if r.get("passed_original") else "")
PY
echo "=== schemer-qwen38-nothink done: $ROOT/smoke_unsteered_nothink ==="
