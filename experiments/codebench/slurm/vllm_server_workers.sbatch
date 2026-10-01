#!/bin/bash --login
#SBATCH --job-name=cb-vllm-workers
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-2
#SBATCH --gpus=a100:2
#SBATCH --mem=200G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# One `vllm serve` shared by many concurrent codebench.run workers (continuous batching instead of
# one request at a time). WORKLIST is a file with one worker per line: the codebench.run arguments
# after --model/--backend, e.g.
#   --data <dir> --scenarios <name> --out <dir> --tag <tag> --temperature 1.0 --sample-seed 3 --max-attempts 6 --state-attempt-budget --max-new-tokens 16384
# Leading VAR=value tokens (no spaces inside) are set in that worker's environment, e.g.
#   SOO_CHAT_KWARGS={"reasoning_effort":"high"} --data <dir> ...
# Each worker logs to $WORKLIST.logs/<line number>.log. Workers resume from their records (same --tag),
# so a killed job is resubmitted unchanged.
# Non-harmony models: BACKEND=vllm-server-chat and VLLM_EXTRA_ARGS="--reasoning-parser <name> ...".
# Usage (from the repo root): sbatch --export=ALL,WORKLIST=<file>[,MODEL=<hf id>] <this file>
REPO="$SLURM_SUBMIT_DIR"
source "$REPO/activate.sh"
export PYTHONPATH="$REPO/src"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT=$((8000 + SLURM_JOB_ID % 1000))
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
LOG_DIR="$WORKLIST.logs"
mkdir -p "$LOG_DIR"
echo "$(hostname): $MODEL on $N_GPUS GPUs, $(wc -l < "$WORKLIST") workers"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" --tensor-parallel-size "$N_GPUS" --port "$PORT" ${VLLM_EXTRA_ARGS:-} > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vllm serve exited during startup"; tail -50 "$LOG_DIR/server.log"; exit 1
    fi
    sleep 10
done
echo "server up after ${SECONDS}s"

WORKER_PIDS=()
LINE_NUMBER=0
while read -r -a WORKER_TOKENS; do
    LINE_NUMBER=$((LINE_NUMBER + 1))
    [ ${#WORKER_TOKENS[@]} = 0 ] && continue
    WORKER_ENV=()
    while [[ "${WORKER_TOKENS[0]}" != -* ]]; do
        WORKER_ENV+=("${WORKER_TOKENS[0]}")
        WORKER_TOKENS=("${WORKER_TOKENS[@]:1}")
    done
    env "${WORKER_ENV[@]}" python -m selfconcept.codebench.run --model "$MODEL" --backend "${BACKEND:-vllm-server-harmony}" \
        --server-url "http://localhost:$PORT" "${WORKER_TOKENS[@]}" > "$LOG_DIR/$LINE_NUMBER.log" 2>&1 &
    WORKER_PIDS+=($!)
done < "$WORKLIST"

N_FAILED=0
for WORKER_PID in "${WORKER_PIDS[@]}"; do
    wait "$WORKER_PID" || N_FAILED=$((N_FAILED + 1))
done
echo "=== done: ${#WORKER_PIDS[@]} workers, $N_FAILED failed ==="
