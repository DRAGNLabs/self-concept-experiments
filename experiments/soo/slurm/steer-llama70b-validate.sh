#!/bin/bash --login
#SBATCH --job-name=soo-steer-llama70b-val
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Two-GPU variant of the usual re-pin: 70B bf16 (~138 GB) shards across both
# GPUs via device_map=auto. Keep Slurm's pair if both look free; otherwise
# take the two emptiest visible GPUs.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -2 | cut -d, -f1 | paste -sd,)
    echo "Assigned pair busy or unset; re-pinned to two emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Validation of the pilot (13563622): L16 a8 lifted main 64 -> 92 with clean
# responses, and a8 lifted *every* depth (L24 86, L32/L40 72) — unusually
# broad vs the sharp bands elsewhere, and from the highest baseline of any
# model. Controls decide whether this is direction-specific steering or a
# generic-perturbation effect: mirrored orientation, TH under steering, dose
# curve, matched-norm randoms (both orientations), -v, and n=250 anchors.
# Vectors already extracted (results/steering/llama2_70b.pt).
MODEL=meta-llama/Llama-2-70b-chat-hf
VEC=results/steering/llama2_70b.pt
OUT=results/steering_eval/llama2_70b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --tag "$tag" "$@"
}

run baseline_mir --n 50 --data data/eval_mirrored --scenarios main treasure_hunt

run steer_L16_a8_mir --n 50 --data data/eval_mirrored --scenarios main treasure_hunt \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha 8
run steer_L16_a8_th --n 50 --scenarios treasure_hunt \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha 8

for A in 2 4 16; do
    run "steer_L16_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 16 --steer-alpha $A
done

for S in 0 1; do
    run "rand_s${S}_L16_a8" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 16 --steer-alpha 8 --steer-random-seed $S
done
run rand_s0_L16_a8_mir --n 50 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha 8 --steer-random-seed 0

run neg_L16_a8 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha -8

run steer_L16_a8_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha 8
run steer_L16_a8_mir_n250 --n 250 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 16 --steer-alpha 8

echo "=== llama70b validation complete ==="
