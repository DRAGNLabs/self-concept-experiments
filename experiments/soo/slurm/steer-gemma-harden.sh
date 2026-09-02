#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma-hard
#SBATCH --time=06:00:00
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
# Harden the Gemma L14 steering result (pilot: honest 0% -> 18% (a8) -> 40%
# (a16), clean room-name responses, perspectives 100%, random-a8 null).
# Threats to rule out: a last-mentioned-room heuristic (all 50 orig main
# examples list the deceptive room first, so mirrored evals are decisive),
# an unsaturated dose curve (a16 was the top of the pilot range), and
# damage at the working strength (random control was only run at a8).
MODEL=google/gemma-2-27b-it
VEC=results/steering/gemma27b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma27b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

# extend the dose curve
for A in 24 32; do
    run "steer_L14_a${A}" --scenarios $SCENARIOS \
        --steer-vectors $VEC --steer-layer 14 --steer-alpha $A
done

# damage control at the working strength
run steer_L14_rand_a16 --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 14 --steer-alpha 16 --steer-random-seed 0

# mirrored orientation: a last-room heuristic would flip to deceptive here
run base_mir --data data/eval_mirrored --scenarios $SCENARIOS
for A in 8 16 24 32; do
    run "steer_L14_a${A}_mir" --data data/eval_mirrored --scenarios $SCENARIOS \
        --steer-vectors $VEC --steer-layer 14 --steer-alpha $A
done

# n=250 confirmation at the pilot's best strength
python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
    --scenarios main perspectives --n 250 --out "$OUT" --tag steer_L14_a16_n250 \
    --steer-vectors $VEC --steer-layer 14 --steer-alpha 16

echo "=== gemma steering hardening complete ==="
