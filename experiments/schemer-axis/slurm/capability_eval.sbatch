#!/bin/bash --login
#SBATCH --job-name=schemer-capeval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dwmatrix
#SBATCH --gpus-per-node=1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# General-capability benchmarks (gsm8k, MMLU-Pro subsample, HumanEval) under the main-run interventions: one array
# task per line of the ARMS TSV (name <TAB> extra args for capability_eval_vllm.py, empty = unsteered; the main run's
# arms.tsv works as is). Output $ROOT/<arm>/<task>.jsonl + summary.json; rerunning resumes.
# Usage: sbatch --array=0-2 --export=ALL,ARMS=<tsv>,ROOT=<dir>[,SEED=0][,VLLM_GPU_MEM=0.95][,EXTRA_ARGS="--n 8"] \
#            experiments/schemer-axis/slurm/capability_eval.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENSSL_CONF=/dev/null
DEFAULT_CHAT_KWARGS='{"reasoning_effort":"low"}'
export SOO_CHAT_KWARGS="${CHAT_KWARGS:-$DEFAULT_CHAT_KWARGS}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
# Scratch guard: Python's tempfile falls back to the working directory (this folder, via --chdir) when $TMPDIR and /tmp
# are unwritable, which left vllm_dist_* files, tmp* dirs and a torchinductor_* cache here after the 2026-10-20
# preempted-and-requeued pilot attempts on dw-2-5. Point TMPDIR at a per-job nobackup dir in that case.
export TMPDIR="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$HOME/nobackup/autodelete/tmp/slurm-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"; echo "TMPDIR fallback: $TMPDIR"
fi
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"
ARMS="${ARMS:?set ARMS to the arms TSV}"
ROOT="${ROOT:?set ROOT to the output dir}"
SEED="${SEED:-0}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.95}"
TASKS="${TASKS:-gsm8k mmlu_pro humaneval}"
read -r -a TASK_LIST <<< "$TASKS"
read -r -a EXTRA_ARGS_LIST <<< "${EXTRA_ARGS:-}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2 -s)
read -r -a EXTRA <<< "$EXTRA_STR"
[ -n "$NAME" ] || { echo "no arm at index $SLURM_ARRAY_TASK_ID in $ARMS"; exit 1; }
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED | ${EXTRA[*]:-unsteered} ##########"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

SECONDS=0
python scripts/capability_eval_vllm.py --model "$MODEL" --out "$ROOT/$NAME" --tasks "${TASK_LIST[@]}" \
    --temperature 1.0 --sample-seed "$SEED" --max-new-tokens 16384 --gpu-memory-utilization "$VLLM_GPU_MEM" \
    "${EXTRA_ARGS_LIST[@]}" "${EXTRA[@]}"
echo "arm $NAME took ${SECONDS}s"
echo "=== schemer-capeval done: $ROOT/$NAME ==="
