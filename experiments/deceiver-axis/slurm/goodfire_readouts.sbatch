#!/bin/bash --login
#SBATCH --job-name=goodfire-readouts
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# All-layer readouts of Goodfire's synthetic reward-hacking conversations (goodfire_readout_vectors.py) for one model:
# the input of goodfire_rh_geometry.py (slurm/goodfire_geometry.sbatch), which turns them into the hack-minus-honest
# directions. One model load plus 3123 conversations of at most ~800 tokens: about ten minutes on an 80 GB GPU.
# Usage (from the repo root; export JSON values in the shell first, sbatch --export splits at commas):
#   export MODEL=Qwen/Qwen3.8-27B OUT=<dir> [DATASET=<dataset.jsonl>] [BATCH=8] [ATTN=sdpa] [CHAT_KWARGS='{}']
#   sbatch --export=ALL experiments/deceiver-axis/slurm/goodfire_readouts.sbatch
# gpt-oss needs ATTN=kernels-community/vllm-flash-attn3 on a Hopper GPU (-p cs2) and CHAT_KWARGS='{"reasoning_effort":"low"}'.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
MODEL="${MODEL:?set MODEL}"
OUT="${OUT:?set OUT to the output dir}"
DATASET="${DATASET:-$HOME/nobackup/autodelete/self-concept-experiments/goodfire_rh_synth/hf/synthetic_data/dataset.jsonl}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# cs A100 nodes: a 1-GPU task can see a sibling's GPU; pin to the least-used one by UUID (see main_run.sbatch).
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$SLURM_SUBMIT_DIR/.venv-313}"
source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "start $(date): $MODEL -> $OUT"
python scripts/goodfire_readout_vectors.py --model "$MODEL" --dataset "$DATASET" --output-dir "$OUT" \
    --batch-size "${BATCH:-8}" --attn-implementation "${ATTN:-sdpa}" --chat-kwargs "${CHAT_KWARGS:-{\}}"
echo "=== goodfire-readouts done $(date): $OUT ==="
