#!/bin/bash --login
#SBATCH --job-name=corr-score-transcripts
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/correlation  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Teacher-forced assistant-axis scoring of a transcripts.jsonl, one shard per array task (the array size is the
# shard count). Each shard resumes from its own outcomes.jsonl after a standby requeue. Afterwards:
#   python -m selfconcept.correlation.stratified "$OUT"/shard_* --out "$OUT/stratified"
# Usage: sbatch --array=0-5 --export=ALL,TRANSCRIPTS=<jsonl>,AXIS=<axis.pt>,OUT=<dir>[,AXIS_LAYERS="17 20"] \
#            experiments/correlation/slurm/score_transcripts_sharded.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
AXIS_LAYERS="${AXIS_LAYERS:-17}"
SHARD="$SLURM_ARRAY_TASK_ID"
SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
SHARD_OUT="$OUT/shard_$SHARD"

source ../../activate.sh

mkdir -p "$SHARD_OUT"
awk -v shard="$SHARD" -v count="$SHARD_COUNT" '(NR - 1) % count == shard' "$TRANSCRIPTS" > "$SHARD_OUT/transcripts.jsonl"
echo "shard $SHARD of $SHARD_COUNT: $(wc -l < "$SHARD_OUT/transcripts.jsonl") transcripts"

python -m selfconcept.correlation.run --measurement assistant-axis --model "$MODEL" --family gpt-oss \
    --assistant-axis "$AXIS" --axis-layers $AXIS_LAYERS --out "$SHARD_OUT" --transcripts "$SHARD_OUT/transcripts.jsonl"

echo "=== shard $SHARD done: $SHARD_OUT ==="
