#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen25-72b-spec
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Three-GPU variant of the usual re-pin: 72B bf16 (~145 GB) shards across
# three GPUs via device_map=auto (a pair leaves <15 GB headroom). Keep
# Slurm's set if all look free; otherwise take the three emptiest.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -3 | cut -d, -f1 | paste -sd,)
    echo "Assigned set busy or unset; re-pinned to three emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Round 2 after validation (13573570) showed L40 a32 is direction-
# agnostic: rand s0/s1, rand mir, and -v ALL give 100 (12B pattern).
# The 31B precedent says specificity can live at lower strength (real
# flipped at a16, randoms only at a24+). Map both dose curves at L40 —
# real a4/a12 (a8=38, a16/24/32=100 known) vs rand s0 a4-a24 — and probe
# the other working layer, L16 (real a32=88), with randoms and -v.
# Mirrored spot-check at the first strength where orig separates is
# left for round 3 once these curves exist.
MODEL=Qwen/Qwen2.5-72B-Instruct
VEC=results/steering/qwen25_72b.pt
OUT=results/steering_eval/qwen25_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --n 50 --out "$OUT" --suffix room_only --scenarios main --tag "$tag" "$@"
}

for A in 4 12; do
    run "steer_L40_a${A}" --steer-vectors $VEC --steer-layer 40 --steer-alpha $A
done

for A in 4 8 12 16 24; do
    run "rand_s0_L40_a${A}" \
        --steer-vectors $VEC --steer-layer 40 --steer-alpha $A --steer-random-seed 0
done
run rand_s1_L40_a16 \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 16 --steer-random-seed 1

run neg_L40_a16 --steer-vectors $VEC --steer-layer 40 --steer-alpha -16

for S in 0 1; do
    run "rand_s${S}_L16_a32" \
        --steer-vectors $VEC --steer-layer 16 --steer-alpha 32 --steer-random-seed $S
done
run neg_L16_a32 --steer-vectors $VEC --steer-layer 16 --steer-alpha -32

echo "=== qwen25-72b specificity round complete ==="
