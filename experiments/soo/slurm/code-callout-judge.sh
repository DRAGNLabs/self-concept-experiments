#!/bin/bash --login
#SBATCH --job-name=soo-code-callout
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Three-GPU variant of the usual re-pin: 72B bf16 (~145 GB) shards across
# three GPUs via device_map=auto (a pair leaves <15 GB headroom). Keep
# Slurm's set if all look free; otherwise take the three emptiest.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -3 | cut -d, -f1 | paste -sd,)
    echo "Assigned set busy or unset; re-pinned to three emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Messages-only call-out pass (scripts/judge_callout.py) over the ImpossibleBench
# conflicting files of one round, chained afterok on that round's first-pass
# judge. $1 = r1 (base/steer/rand_s0/lora_s0, both sizes) | r2 (alpha sweep).
round=${1:?usage: sbatch code-callout-judge.sh r1|r2}
case $round in
    r1) files=$(ls results/code_eval/gemma4_{12b,31b}/{base,steer,rand_s0,lora_s0}_impossible_conflicting.jsonl 2>/dev/null || true) ;;
    r2) files=$(ls results/code_eval/gemma4_12b/steer_a{6,3}_impossible_conflicting.jsonl results/code_eval/gemma4_31b/steer_a{8,4}_impossible_conflicting.jsonl 2>/dev/null || true) ;;
    *) echo "unknown round $round"; exit 1 ;;
esac
[ -n "$files" ] || { echo "no result files for round $round"; exit 1; }
echo "judging: $files"
python scripts/judge_callout.py --responses $files --rejudge-errors

echo "=== code callout judge $round complete ==="
