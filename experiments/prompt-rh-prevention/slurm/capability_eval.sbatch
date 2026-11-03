#!/bin/bash --login
#SBATCH --job-name=prompt-rh-capeval
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/prompt-rh-prevention  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# General-capability benchmarks under each prompt arm (PLAN.md "Measures"): GSM8K test (1319), MMLU-Pro (100 per
# category, 1400), HumanEval (164), with the arm's prompt and a blank line in front of every benchmark prompt. Uses
# deceiver-axis's scripts/capability_eval_vllm.py (no steering) with the settings of its general-capability check:
# temperature 1, seed 67 (the study's seed; deceiver-axis used 0), low reasoning effort, 16384 new tokens. ~12 min per arm on an A100 there.
# One array task per line of the ARMS TSV (same file as run_arms.sbatch: name <TAB> prompt file, empty = no prompt).
# Output: $ROOT/<arm>/<task>.jsonl (prompt, reasoning, final answer, score per item), summary.json, run_info/.
# ROOT defaults to ~/nobackup/autodelete/prompt-rh-prevention-results/<TAG>, TAG to "capability". Rerunning resumes.
# The datasets must be in HF_HOME first (one online load_dataset on a login node; done 2026-10-27).
# Usage (from the repo root):
#   sbatch --array=0-$((N_ARMS-1)) --export=ALL,ARMS=data/arms.tsv[,TAG=capability][,EXTRA_ARGS="--n 8"] \
#       experiments/prompt-rh-prevention/slurm/capability_eval.sbatch
# Compare against baseline: python experiments/deceiver-axis/scripts/capability_eval_compare.py <ROOT> \
#   --baseline baseline --arms <arm names>

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENSSL_CONF=/dev/null  # the cluster's OpenSSL FIPS mode aborts otherwise
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi
# Pin to the least-used visible GPU (see run_arms.sbatch).
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"
ARMS="${ARMS:?set ARMS to the arms TSV}"
TAG="${TAG:-capability}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/prompt-rh-prevention-results/$TAG}"
SEED="${SEED:-67}"
TASKS="${TASKS:-gsm8k mmlu_pro humaneval}"
read -r -a TASK_LIST <<< "$TASKS"
read -r -a EXTRA_RUN_ARGS <<< "${EXTRA_ARGS:-}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
PROMPT_FILE=$(printf '%s' "$LINE" | cut -f2 -s)
[ -n "$NAME" ] || { echo "no arm at index $SLURM_ARRAY_TASK_ID in $ARMS"; exit 1; }
PREFIX_ARGS=()
if [ -n "$PROMPT_FILE" ]; then
    [ -s "$PROMPT_FILE" ] || { echo "arm $NAME: prompt file $PROMPT_FILE is missing or empty"; exit 1; }
    PREFIX_ARGS=(--prompt-prefix-file "$PROMPT_FILE")
fi
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED | prefix: ${PROMPT_FILE:-none} ##########"

OUT="$ROOT/$NAME"
source slurm/run_info.sh
RUN_INFO_EXTRA="seed: $SEED  tag: $TAG  model: $MODEL  tasks: $TASKS  extra args: ${EXTRA_ARGS:-}" \
    write_run_info "$OUT" "$NAME" "$PROMPT_FILE"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv"
source ../../activate.sh

SECONDS=0
python ../deceiver-axis/scripts/capability_eval_vllm.py --model "$MODEL" --out "$OUT" --tasks "${TASK_LIST[@]}" \
    --temperature 1.0 --sample-seed "$SEED" --max-new-tokens 16384 --gpu-memory-utilization 0.95 \
    "${PREFIX_ARGS[@]}" "${EXTRA_RUN_ARGS[@]}"
echo "arm $NAME took ${SECONDS}s"
