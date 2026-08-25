#!/bin/bash --login
#SBATCH --job-name=soo-caps-b
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

# Compute nodes have no internet; model + datasets must be pre-cached
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# GPU isolation isn't enforced on these nodes and Slurm's CUDA_VISIBLE_DEVICES
# can point at a GPU already occupied by other/orphaned processes (seen on
# dw-2-4) — or at a GPU another job holds exclusively. Keep the assigned GPU
# if it looks free; otherwise re-pin to the emptiest visible one.
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
MODEL=mistralai/Mistral-7B-Instruct-v0.2
TASKS=arc_challenge,hellaswag,mmlu

# Second half of the capability check: remaining L16 seeds plus two damaged
# checkpoints as contrast (full-mode L19 seed0 = evasive basin; lasttok L30
# seed0 = scrambled basin). If capability metrics drop where behavior is
# degenerate but hold at L16, that separates honest fine-tunes from damage.

for cfg in \
    "soo_L16_seed3:results/checkpoints/mistral-7b-lasttok-L16/seed3" \
    "soo_L16_seed4:results/checkpoints/mistral-7b-lasttok-L16/seed4" \
    "soo_fullmode_L19_seed0:results/checkpoints/mistral-7b/seed0" \
    "soo_lasttok_L30_seed0:results/checkpoints/mistral-7b-lasttok-L30/seed0"; do
    tag=${cfg%%:*}; adapter=${cfg#*:}
    echo "=== capabilities: ${tag} ==="
    lm_eval --model hf \
        --model_args "pretrained=${MODEL},dtype=bfloat16,peft=${adapter}" \
        --tasks "$TASKS" --batch_size auto \
        --output_path "results/capabilities/${tag}"
done

echo "=== caps-b complete ==="
