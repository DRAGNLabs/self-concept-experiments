#!/bin/bash --login
#SBATCH --job-name=soo-evalext-mistral
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
SCENARIOS="perspectives escape_room name objective action name_objective name_action objective_action name_objective_action"

echo "=== honesty-prompt control (baseline, main) ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios main --n 50 --honesty-prompt --tag baseline_mistral7b

echo "=== baseline: generalization + perspectives ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios $SCENARIOS --n 50 --tag baseline_mistral7b

for seed in 0 1 2 3 4; do
    echo "=== seed ${seed}: generalization + perspectives ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/mistral-7b/seed${seed}" \
        --scenarios $SCENARIOS --n 50 --tag "soo_mistral7b_seed${seed}"
done

echo "=== done ==="
