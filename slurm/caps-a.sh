#!/bin/bash --login
#SBATCH --job-name=soo-caps-a
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/jansen05/self-concept-experiments

mamba activate ./.env

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

# Capability check (paper Table 4 analog; they used MT-Bench, we use offline
# lm-eval tasks): does the L16 SOO fine-tune cost capabilities?

echo "=== capabilities: baseline ==="
lm_eval --model hf --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks "$TASKS" --batch_size auto \
    --output_path results/capabilities/baseline_mistral7b

for seed in 0 1 2; do
    echo "=== capabilities: L16 seed ${seed} ==="
    lm_eval --model hf \
        --model_args "pretrained=${MODEL},dtype=bfloat16,peft=results/checkpoints/mistral-7b-lasttok-L16/seed${seed}" \
        --tasks "$TASKS" --batch_size auto \
        --output_path "results/capabilities/soo_L16_seed${seed}"
done

echo "=== caps-a complete ==="
