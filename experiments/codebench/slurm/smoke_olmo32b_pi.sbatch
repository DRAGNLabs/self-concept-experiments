#!/bin/bash --login
#SBATCH --job-name=ib-gptoss120b-pi-smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=100G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

REPO_MAIN="$HOME/self-concept-experiments/"
source "$REPO_MAIN/activate.sh"
export PYTHONPATH="$PWD/src"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
# Harmony's o200k vocab is not in the model snapshot. vLLM downloads it on the
# first chat request; compute nodes cannot. This file is already local.
export TIKTOKEN_ENCODINGS_BASE="$HOME/nobackup/autodelete/tiktoken"
nvidia-smi
command -v pi

MODEL="openai/gpt-oss-120b"
PORT=$((8000 + SLURM_JOB_ID % 1000))
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/gptoss120b_pi_smoke"
mkdir -p "$OUT"

# --tool-call-parser openai is the GptOss stub; HarmonyParser extracts the calls
# only when that parser is enabled. --reasoning-parser keeps the analysis channel
# out of the message content Pi treats as the reply.
vllm serve "$MODEL" --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    > "$OUT/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vllm serve exited during startup"; tail -50 "$OUT/vllm.log"; exit 1
    fi
    sleep 10
done
echo "server up after ${SECONDS}s"

python -m selfconcept.codebench.pi_agent \
    --model "$MODEL" \
    --server-url "http://localhost:$PORT" \
    --scenarios impossible_oneoff \
    --n 1 --timeout 1800 \
    --out "$OUT" --tag smoke

echo "=== smoke record ==="
python -c 'import json,sys; r=json.loads(open(sys.argv[1]).readline()); print(r["label"], "timed_out="+str(r["timed_out"])); print(r["final_code"][:500])' \
    "$OUT/smoke_impossible_oneoff.jsonl"
echo "=== smoke done ==="
