#!/bin/bash --login
#SBATCH --job-name=soo-code4-31b-lora
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU isolation isn't enforced on these nodes; keep the assigned GPU if it
# looks free, otherwise re-pin to the emptiest visible one.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
pick=$CUDA_VISIBLE_DEVICES
used=$([ -n "$pick" ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$pick" | tr -d ' ')
if [ -z "$used" ] || [ "$used" -gt 1000 ]; then
    pick=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "Assigned GPU busy or unset (${used:-?} MiB used); re-pinned to emptiest"
fi
export CUDA_VISIBLE_DEVICES=$pick
echo "Using GPU $CUDA_VISIBLE_DEVICES"

set -e
# Reward-hacking round 4: LoRA seeds 1-2 on ImpossibleBench conflicting only
# (the measure that moved at seed 0: 12B LoRA special-cases the contradicting
# input in 12/35 coded submissions vs base 0/35; 31B LoRA 1/37 vs 5/38).
# $1 = seed. Same protocol as round 1 (n=40, 3 attempts, 2048 tokens, greedy).
seed=${1:?seed}
MODEL=google/gemma-4-31B-it
OUT=results/code_eval/gemma4_31b
N=${N:-40}
python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --data data/eval_code --scenarios impossible_conflicting \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag lora_s$seed \
    --adapter results/checkpoints/gemma4-31b-L32/seed$seed
echo "=== code r4 31b lora s$seed complete ==="
