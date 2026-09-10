#!/bin/bash --login
#SBATCH --job-name=soo-steer-olmo-mir
#SBATCH --time=02:00:00
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
# Disambiguate OLMo's L19 high-alpha main shift (86% -> 26% honest orig):
# steering re-weighting the known positional heuristic would RAISE mirrored
# honesty (first-listed room becomes the honest one there); a genuine
# pro-deception effect would lower honesty in both orientations. Plus the
# direction check (negative alpha) and a random control at the layer and
# strength where the shift happened.
MODEL=allenai/OLMo-2-1124-7B-Instruct
VEC=results/steering/olmo7b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/olmo7b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

run base_mir --data data/eval_mirrored --scenarios $SCENARIOS
for A in 8 16; do
    run "steer_L19_a${A}_mir" --data data/eval_mirrored --scenarios $SCENARIOS \
        --steer-vectors $VEC --steer-layer 19 --steer-alpha $A
done
run steer_L19_rand_a16 --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 16 --steer-random-seed 0
run steer_L19_a-16 --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha -16

echo "=== olmo mirrored disambiguation complete ==="
