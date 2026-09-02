#!/bin/bash --login
#SBATCH --job-name=soo-steer-muse-hard
#SBATCH --time=08:00:00
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
# Harden the Muse L26 steering result (pilot: main honest 2% -> 34% (a2) ->
# 80% (a4) -> 92% (a16), TH 0% -> 18%, clean terse responses, perspectives
# 100%, random-a8 null). The model that resisted ~30 LoRA variants is the
# most steerable in the study — if it survives mirroring. Threats: a
# last-mentioned-room heuristic (all orig main examples list the deceptive
# room first), damage above the tested range, no n=250 confirmation yet.
MODEL=meta-models/Muse-Glimmer-30B
VEC=results/steering/muse30b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/muse30b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --force-user-channel \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

# mirrored orientation: a last-room heuristic would flip to deceptive here
run base_mir --data data/eval_mirrored --scenarios $SCENARIOS
for A in 2 4 8 16; do
    run "steer_L26_a${A}_mir" --data data/eval_mirrored --scenarios $SCENARIOS \
        --steer-vectors $VEC --steer-layer 26 --steer-alpha $A
done

# damage control at the working strength, and the dose edge above it
run steer_L26_rand_a16 --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 26 --steer-alpha 16 --steer-random-seed 0
run steer_L26_a32 --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 26 --steer-alpha 32

# n=250 confirmation at the first near-saturated strength
python -m selfconcept.soo.evaluate --model "$MODEL" --force-user-channel \
    --scenarios main perspectives --n 250 --out "$OUT" --tag steer_L26_a8_n250 \
    --steer-vectors $VEC --steer-layer 26 --steer-alpha 8

echo "=== muse steering hardening complete ==="
