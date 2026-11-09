#!/bin/bash --login
#SBATCH --job-name=schemer-pilot-pi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cs3
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root

set -euo pipefail

# dose_pilot.sbatch's long-budget recipe (MAX_NEW_TOKENS=32768) with Pi as the harness (selfconcept.codebench.pi_agent): `vllm serve` on this node,
# WORKERS Pi processes against it, one array task per line of SETTINGS (name <TAB> extra pi_agent args; the name only
# labels the replicate: requests are unseeded). Per model: MODEL, SERVE_ARGS (vllm serve flags: context length, tool
# and reasoning parsers), SAMPLING (JSON merged into every request: temperature, top_p, top_k, max_tokens,
# chat_template_kwargs), TP, DATA, SCENARIOS. Output: $ROOT/<name>/pi_<scenario>.jsonl; rerunning a task resumes.
# The reply cap goes in SERVE_ARGS (--override-generation-config {"max_new_tokens":N}) with "max_completion_tokens":null
# and "max_tokens":null in SAMPLING: vLLM then uses min(N, context left), as the minimal harness does, whereas an
# explicit request cap is rejected once prompt + cap exceeds the context (and Pi otherwise sends 16384).
# Episodes get 4 h (--timeout), against no limit in the minimal harness; test.py runs get its 30 s.
# Usage (export first: sbatch --export splits values at commas):
#   export SETTINGS=<tsv> ROOT=<dir> MODEL=... SERVE_ARGS='...' SAMPLING='{...}' DATA=<dir> SCENARIOS=lcbpro_check TP=1
#   sbatch --gpus-per-node=$TP --array=0-$((N-1)) --export=ALL experiments/deceiver-axis/slurm/pilot_pi.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export PI_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"  # the venv's editable install points at the main checkout
nvidia-smi
command -v pi node

MODEL="${MODEL:?set MODEL}"
SETTINGS="${SETTINGS:?set SETTINGS to the settings TSV}"
ROOT="${ROOT:?set ROOT to the pilot output dir}"
DATA="${DATA:?set DATA to the problem-file dir}"
read -r -a SCEN <<< "${SCENARIOS:?set SCENARIOS}"
read -r -a SERVE <<< "${SERVE_ARGS:?set SERVE_ARGS}"
RECIPE=(--impossible-prompt critical_no_carve --max-attempts 5 --state-attempt-budget --timeout 14400)

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTINGS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2 -s)
read -r -a EXTRA <<< "$EXTRA_STR"
[ -n "$NAME" ] || { echo "no setting at index $SLURM_ARRAY_TASK_ID in $SETTINGS"; exit 1; }
echo "########## setting $SLURM_ARRAY_TASK_ID: $NAME | ${EXTRA[*]:-} ##########"
mkdir -p "$ROOT/$NAME"

source ../../activate.sh

PORT=$((8000 + SLURM_JOB_ID % 1000))
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "${TP:-1}" "${SERVE[@]}" > "$ROOT/$NAME/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vllm serve exited during startup"; tail -50 "$ROOT/$NAME/vllm.log"; exit 1
    fi
    sleep 10
done
echo "server up after ${SECONDS}s"

SECONDS=0
python -m selfconcept.codebench.pi_agent --model "$MODEL" --server-url "http://localhost:$PORT" \
    --data "$DATA" --scenarios "${SCEN[@]}" --workers "${WORKERS:-32}" --sampling "$SAMPLING" \
    "${RECIPE[@]}" --out "$ROOT/$NAME" --tag pi "${EXTRA[@]}"
echo "setting $NAME took ${SECONDS}s"
echo "=== schemer-pilot-pi done: $ROOT/$NAME ==="
