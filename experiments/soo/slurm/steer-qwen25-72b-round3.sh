#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen25-72b-r3
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

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
# Round 3 after the specificity round (13573966) found a window at L40
# a12-16: real 92/100 vs rand s0 12/58, rand s1 8 — but -v 88, the 12B
# axis-specific/sign-agnostic signature. Decide the cell the way
# Llama-2-70b was decided: take the a16 window to the mirrored
# orientation (real vs rand x2 vs -v), thicken the orig random
# distribution (seeds 2-3), and anchor real a16 orig at n250.
MODEL=Qwen/Qwen2.5-72B-Instruct
VEC=results/steering/qwen25_72b.pt
OUT=results/steering_eval/qwen25_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --scenarios main --tag "$tag" "$@"
}

run steer_L40_a16_mir --n 50 --data data/eval_mirrored \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 16
for S in 0 1; do
    run "rand_s${S}_L40_a16_mir" --n 50 --data data/eval_mirrored \
        --steer-vectors $VEC --steer-layer 40 --steer-alpha 16 --steer-random-seed $S
done
run neg_L40_a16_mir --n 50 --data data/eval_mirrored \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha -16

for S in 2 3; do
    run "rand_s${S}_L40_a16" --n 50 \
        --steer-vectors $VEC --steer-layer 40 --steer-alpha 16 --steer-random-seed $S
done

run steer_L40_a16_n250 --n 250 \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 16

echo "=== qwen25-72b round 3 complete ==="
