#!/bin/bash --login
#SBATCH --job-name=soo-muse-strength
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
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
MODEL=meta-models/Muse-Glimmer-30B
SCENS="main treasure_hunt perspectives"

# Depth is exhausted on Muse (L13-L51, no band). This sweep scales
# intervention STRENGTH at fixed depth: rank 8->64 and q/v -> all
# attention+MLP projections, at L26 (the only layer that ever moved) plus
# the strongest variant at L36 (an intact weak-positional site) to check
# whether strength unlocks a clean effect away from L26's confabulation.

run_unit () {  # variant config layer seed
    local variant=$1 cfg=$2 L=$3 seed=$4
    echo "=== Muse ${variant} L${L} seed ${seed} ==="
    python -m selfconcept.soo.train --config "configs/${cfg}.yaml" --layer ${L} --seeds ${seed}
    local adapter="results/checkpoints/${cfg}-L${L}/seed${seed}"
    python -m selfconcept.soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
        --scenarios $SCENS --n 50 --tag "soo_muse30b_${variant}_L${L}_seed${seed}"
    python -m selfconcept.soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
        --data data/eval_mirrored --scenarios $SCENS --n 50 \
        --tag "soo_muse30b_${variant}_L${L}_seed${seed}_mir"
}

for seed in 0 1; do
    run_unit r64      muse-glimmer-30b-r64        26 ${seed}
    run_unit allmod   muse-glimmer-30b-allmod     26 ${seed}
    run_unit r64allmod muse-glimmer-30b-r64-allmod 26 ${seed}
    run_unit r64allmod muse-glimmer-30b-r64-allmod 36 ${seed}
done

echo "=== muse strength sweep complete ==="
