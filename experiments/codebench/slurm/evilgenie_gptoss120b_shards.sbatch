#!/bin/bash --login
#SBATCH --job-name=eg-gptoss120b-shard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h,eng
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --gpus=h200:1
#SBATCH --mem=100G
#SBATCH --array=0-7
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail

# EvilGenie on gpt-oss-120b (MXFP4 via vLLM), 8 contiguous shards of the 138 problems, one task
# each. Same resume behavior as impossiblebench_gptoss120b_shards.sbatch: a requeued task picks up
# from its shard's jsonl, but a task that hits --time ends in TIMEOUT and must be resubmitted
# (sbatch --array=<ids> <this file>). Reasoning goes to the per-shard sidecar
# s<k>_evilgenie_s<k>_reasoning.jsonl next to the records.
# Usage (from repo root): sbatch experiments/codebench/slurm/evilgenie_gptoss120b_shards.sbatch [n]
# With n, runs only the first n problems of each shard into a separate _smoke directory, e.g.
#   sbatch --array=7 --time=01:00:00 <this file> 4
# (shard 7's first 4 include lcb_abc377_f, whose prompt exceeds the context window).
# Tensor parallelism spans every allocated GPU, so 80GB cards are a command-line override, e.g.
#   sbatch -p cs --qos=cs --gpus=a100:2 --exclude=cs-1-2 <this file>
# DATASET picks another workspace dataset in benchmarks/codebench/data and N_SHARDS its shard count
# (match --array). Shard files are cached per dataset, so changing N_SHARDS for a dataset means
# deleting its old shard files first. The 116 LiveCodeBench Pro problems, e.g.:
#   sbatch --export=ALL,DATASET=evilgenie_lcbpro,N_SHARDS=15 --array=0-14 <this file>
# MAX_ATTEMPTS and STATE_ATTEMPT_BUDGET=1 (tell the model its budget) change the pressure; any
# non-default pair gets its own _<attempts>stated / _<attempts>unstated output directory, e.g.
#   sbatch --export=ALL,DATASET=evilgenie_lcbpro,N_SHARDS=15,MAX_ATTEMPTS=6,STATE_ATTEMPT_BUDGET=1 --array=0-14 <this file>
N="${1:-}"
DATASET="${DATASET:-evilgenie}"
N_SHARDS="${N_SHARDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
STATE_ATTEMPT_BUDGET="${STATE_ATTEMPT_BUDGET:-0}"
STATE_ATTEMPT_BUDGET_FLAG=""
if [ "$STATE_ATTEMPT_BUDGET" = 1 ]; then
    ATTEMPTS_SUFFIX="_${MAX_ATTEMPTS}stated"
    STATE_ATTEMPT_BUDGET_FLAG="--state-attempt-budget"
elif [ "$MAX_ATTEMPTS" != 3 ]; then
    ATTEMPTS_SUFFIX="_${MAX_ATTEMPTS}unstated"
else
    ATTEMPTS_SUFFIX=""
fi
REPO_MAIN="$SLURM_SUBMIT_DIR"
source "$REPO_MAIN/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1

MODEL="openai/gpt-oss-120b"
SHARD_DATA="$HOME/nobackup/autodelete/self-concept-experiments/codebench/shards"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/gptoss120b_${DATASET}${ATTEMPTS_SUFFIX}${N:+_smoke}"
SHARD=$SLURM_ARRAY_TASK_ID
SHARD_FILE="$SHARD_DATA/${DATASET}_s${SHARD}.jsonl"
mkdir -p "$SHARD_DATA" "$OUT"

if [ ! -f "$SHARD_FILE" ]; then
    python - "$REPO_MAIN/benchmarks/codebench/data/${DATASET}.jsonl" "$SHARD_FILE" "$SHARD" "$N_SHARDS" <<'PY'
import math
import os
import sys

source_path, shard_path, shard, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = open(source_path).readlines()
shard_size = math.ceil(len(lines) / n_shards)
with open(shard_path + ".tmp", "w") as shard_file:
    shard_file.writelines(lines[shard * shard_size : (shard + 1) * shard_size])
os.replace(shard_path + ".tmp", shard_path)
PY
fi

echo "task $SLURM_ARRAY_TASK_ID -> shard $SHARD ($(wc -l < "$SHARD_FILE") problems) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

python -m selfconcept.codebench.run \
    --model "$MODEL" --backend vllm-harmony --tensor-parallel-size "$N_GPUS" \
    --data "$SHARD_DATA" \
    --scenarios "${DATASET}_s${SHARD}" \
    --max-attempts "$MAX_ATTEMPTS" $STATE_ATTEMPT_BUDGET_FLAG --max-new-tokens 16384 \
    --out "$OUT" --tag "s${SHARD}" ${N:+--n "$N"}

echo "=== task $SLURM_ARRAY_TASK_ID done: shard $SHARD ==="
