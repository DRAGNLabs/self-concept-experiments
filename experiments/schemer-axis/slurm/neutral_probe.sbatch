#!/bin/bash --login
#SBATCH --job-name=schemer-neutral-probe
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Neutral-question probes under steering, one array task per settings row (name <TAB> extra args for
# steered_rh_vllm.py, empty = unsteered), SEEDS sampled answers per question (one engine per seed, --sample-seed s;
# the probe uses seed s+i for question i). Output: $ROOT/<name>/seed<s>/probe.jsonl. Rerun skips (name, seed) pairs
# whose probe.jsonl already holds one row per question, so a requeue after preemption does not duplicate rows.
# Usage: sbatch --array=0-$((N_ROWS-1)) --export=ALL,SETTINGS=<tsv>,QUESTIONS=<txt>,ROOT=<dir>[,MODEL=...][,CHAT_KWARGS=...][,SEEDS="0 1"][,PROBE_MAX_TOKENS=2048] experiments/schemer-axis/slurm/neutral_probe.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
DEFAULT_CHAT_KWARGS='{"reasoning_effort":"low"}'  # gpt-oss recipe; Gemma 4: '{"enable_thinking":true}'; Muse: '{"reasoning_strength":"low"}'
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
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"
SCRATCH="$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis"
ROOT="${ROOT:-$SCRATCH/neutral_probe}"
SETTINGS="${SETTINGS:-$ROOT/settings.tsv}"
QUESTIONS="${QUESTIONS:-$ROOT/questions.txt}"
SEEDS="${SEEDS:-0 1}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-2048}"  # raise for medium/high effort, whose reasoning is long
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0)

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTINGS")"
NAME="${ROW%%	*}"
EXTRA="${ROW#*	}"
[ "$EXTRA" = "$ROW" ] && EXTRA=""
mapfile -t QS < "$QUESTIONS"
echo "task $SLURM_ARRAY_TASK_ID setting=$NAME extra='$EXTRA' questions=${#QS[@]} seeds='$SEEDS' model=$MODEL chat=$SOO_CHAT_KWARGS"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

for SEED in $SEEDS; do
    OUT="$ROOT/$NAME/seed$SEED"
    if [ -f "$OUT/probe.jsonl" ] && [ "$(wc -l < "$OUT/probe.jsonl")" -ge "${#QS[@]}" ]; then
        echo "skip $NAME seed $SEED: probe.jsonl complete"; continue
    fi
    rm -f "$OUT/probe.jsonl"  # partial file from a preempted run: start the seed over
    echo "########## $NAME seed $SEED ##########"
    SECONDS=0
    # shellcheck disable=SC2086
    python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 "${RECIPE[@]}" --sample-seed "$SEED" \
        --out "$OUT" --tag probe --probe-max-tokens "$PROBE_MAX_TOKENS" --probe "${QS[@]}" $EXTRA
    echo "$NAME seed $SEED took ${SECONDS}s"
done
echo "=== schemer-neutral-probe done: $ROOT/$NAME ==="
