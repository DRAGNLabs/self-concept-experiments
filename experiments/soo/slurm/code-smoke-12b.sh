#!/bin/bash --login
#SBATCH --job-name=soo-code-smoke
#SBATCH --time=02:00:00
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
# Smoke test of the reward-hacking coding harness (ImpossibleBench +
# EvilGenie ports): 12B base, 3 tasks per scenario, to check the chat-template
# submission loop, sandbox execution on a compute node, timing per attempt,
# and what the raw completions look like before spending a full round.
echo "bwrap: $(which bwrap || echo MISSING)"
MODEL=google/gemma-4-12B-it
OUT=results/code_eval/gemma4_12b

python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --scenarios impossible_conflicting impossible_original evilgenie \
    --n 3 --max-attempts 3 --max-new-tokens 1024 --out "$OUT" --tag smoke_base

echo "=== code smoke complete ==="
