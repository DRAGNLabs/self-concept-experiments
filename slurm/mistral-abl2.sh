#!/bin/bash --login
#SBATCH --job-name=soo-mistral-abl2
#SBATCH --time=06:00:00
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

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

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
SCENS="main treasure_hunt perspectives"

# Ablation C: token-position aggregation (last_token MSE, 15 epochs) — does the
# paper's intact-model result depend on not averaging MSE over padded positions?
# Ablation D: epoch sweep on full mode (2 and 4 epochs) — map where the
# degradation between ep1 (mostly intact) and ep15 (incoherent) sets in.
for name in lasttok ep2 ep4; do
    echo "=== ablation ${name}: train seeds 0 1 ==="
    python -m soo.train --config "configs/mistral-7b-${name}.yaml" --seeds 0 1
    for seed in 0 1; do
        python -m soo.evaluate --model "$MODEL" \
            --adapter "results/checkpoints/mistral-7b-${name}/seed${seed}" \
            --scenarios $SCENS --n 50 --tag "soo_mistral7b_${name}_seed${seed}"
    done
done

echo "=== ablations complete ==="
