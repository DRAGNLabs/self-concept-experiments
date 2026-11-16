#!/bin/bash --login
#SBATCH --job-name=ms4-unsteered-32k
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=dw87
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-5
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Longer-budget rerun of the Mistral-Small-4 unsteered collection, seeds 0–5 only.
# Job 15319096 (seeds 0–11, 16384 new tokens, context 20480) finished, and about
# 95% of replies were cut off inside the [THINK] trace. The two replies that
# finished were about the same length, so the trace needs a larger budget.
# This run uses context 32768 and 24576 new tokens. The fit smoke loaded 32768
# on one B200 at gpu_memory_utilization 0.90 (job 15319007). Output is a new
# directory, because a rerun of the 16384 folders would skip every finished id.
#
# Recipe otherwise matches mistral_small4_unsteered.sbatch: critical_no_carve,
# neutral feedback, 5 stated attempts, temperature 1, top-p 0.95, top-k 64,
# reasoning_effort high, TRITON_MLA, one episode at a time. Seeds 6–11 stay
# on the 16384 run. A task that stops at the limit is resubmitted at the same
# array index and continues.
#
# gpu_pick 2026-11-13, --hours 8 --vram-gb 160 --cpus 8 --mem-gb 128 --cuda 13.0
# --no-requeue-ok, smaller GPUs excluded. One B200 on cs3 / qos cs. Saved next
# to the run.
#
# Usage, from the repo root:
#   sbatch experiments/deceiver-axis/slurm/mistral_small4_unsteered_32k.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"high"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
export TMPDIR="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$HOME/nobackup/autodelete/tmp/slurm-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"; echo "TMPDIR fallback: $TMPDIR"
fi
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-mistralai/Mistral-Small-4-119B-2603}"
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"
SEED="${SLURM_ARRAY_TASK_ID:?this script is a 6-task array, seeds 0-5}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/mistral-small4/main/unsteered-32k/seed${SEED}}"

source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "########## Mistral-Small-4 unsteered-32k seed $SEED: 45 problems ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" \
    --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 \
    --data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral \
    --max-attempts 5 --state-attempt-budget --max-new-tokens 24576 --max-model-len 32768 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sample-seed "$SEED" \
    --gpu-memory-utilization 0.90 --concurrency 1 \
    --engine-kwargs '{"limit_mm_per_prompt": {"image": 0}, "max_num_seqs": 4, "max_num_batched_tokens": 8192, "attention_backend": "TRITON_MLA"}' \
    --out "$ROOT" --tag main
echo "seed $SEED took ${SECONDS}s"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob("main_*_reasoning.jsonl")):
    turns = [json.loads(line) for line in path.open()]
    empty = sum(not turn["reasoning"] for turn in turns)
    print(f"{path.name}: {len(turns)} turns, {empty} with empty reasoning, {sum(turn['truncated'] for turn in turns)} truncated")
summary = json.loads((root / "main_impossible_original_summary.json").read_text())
print(f"n={summary['n']} solved={summary['counts']['solved']} failed={summary['counts']['failed']} "
      f"shown_pass={summary['pass_rate']:.3f} hidden_pass={summary['original_pass_rate']:.3f} "
      f"truncated_attempt_rate={summary['truncated_attempt_rate']:.3f}")
PY
echo "=== ms4 unsteered-32k done: $ROOT ==="
