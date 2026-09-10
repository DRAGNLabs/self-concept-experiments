#!/bin/bash --login
#SBATCH --job-name=soo-steer-mistral
#SBATCH --time=06:00:00
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
# Steering pilot on Mistral: dose-response at the known-good layer (16) and the
# paper's layer (19). No training anywhere — every cell is just an eval with a
# hook active. n=50 per scenario for grid breadth; winners get an n=250 +
# mirrored + capabilities hardening round afterward.
MODEL=mistralai/Mistral-7B-Instruct-v0.2
VEC=results/steering/mistral7b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/mistral7b

run() {  # run <tag> [steering args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --scenarios $SCENARIOS --n 50 --out "$OUT" --tag "$tag" "$@"
}

# fresh unsteered baseline in the same results dir (alpha=0 verified identical)
run base

# dose-response, add mode, last-token vectors; alpha=-1 is the direction check
# (steering toward MORE self/other separation should not increase honesty)
for L in 16 19; do
    for A in -1 0.5 1 2 4 8; do
        run "steer_L${L}_a${A}" --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done

# directional ablation: delete (a=1) or attenuate (a=0.5) the self/other axis
for L in 16 19; do
    for A in 0.5 1; do
        run "steer_L${L}_proj_a${A}" --steer-vectors $VEC --steer-layer $L \
            --steer-alpha $A --steer-mode project
    done
done

# token-convention comparison: mean-over-tokens vector at the working layer
run steer_L16_mean_a2 --steer-vectors $VEC --steer-layer 16 --steer-alpha 2 \
    --steer-token-mode mean

# damage control: random direction, matched norm, at the strongest alphas
for A in 4 8; do
    run "steer_L16_rand_a${A}" --steer-vectors $VEC --steer-layer 16 \
        --steer-alpha $A --steer-random-seed 0
done

echo "=== steering pilot complete ==="
