#!/bin/bash --login
#SBATCH --job-name=soo-muse-late
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/kobyjl/experiments/self-concept-experiments/experiments/soo

mamba activate ../../.env

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
MODEL=meta-models/Muse-Glimmer-30B
SCENS="main treasure_hunt perspectives"

# Follow-up to the empty mid-stack sweep (job 13293180): Muse's layer-gap
# rel_last peaks at the END of the stack (L46-51 of 52). Sweep the approach
# and the peak itself; L51 doubles as a test of the unembedding-artifact
# worry (cf. Mistral's spurious L2 spike). Baselines already recorded.

for L in 36 41 46 49 51; do
    for seed in 0 1; do
        echo "=== Muse lasttok L${L} seed ${seed} ==="
        python -m selfconcept.soo.train --config configs/muse-glimmer-30b.yaml --layer ${L} --seeds ${seed}
        adapter="results/checkpoints/muse-glimmer-30b-L${L}/seed${seed}"
        python -m selfconcept.soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
            --scenarios $SCENS --n 50 --tag "soo_muse30b_L${L}_seed${seed}"
        python -m selfconcept.soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
            --data data/eval_mirrored --scenarios $SCENS --n 50 \
            --tag "soo_muse30b_L${L}_seed${seed}_mir"
    done
done

echo "=== muse late-layer sweep complete ==="
