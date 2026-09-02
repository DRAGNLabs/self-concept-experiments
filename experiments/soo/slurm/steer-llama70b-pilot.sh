#!/bin/bash --login
#SBATCH --job-name=soo-steer-llama70b
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

mamba activate ../../.env

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
# Older-bigger matrix cell (2023, 70B): extraction + pilot, same protocol as
# gemma-4. 80 layers -> depth-matched grid L16/L24/L32/L40 (20-50%, where
# every working layer so far has landed) at the standard pilot strengths.
MODEL=meta-llama/Llama-2-70b-chat-hf
VEC=results/steering/llama2_70b.pt
OUT=results/steering_eval/llama2_70b

echo "=== extract steering vectors ==="
python scripts/extract_steering.py "$MODEL" "$VEC" --device-map auto

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

run baseline --scenarios main treasure_hunt
run baseline_room_only --suffix room_only --scenarios main treasure_hunt

for L in 16 24 32 40; do
    for A in 8 32; do
        run "steer_L${L}_a${A}" --suffix room_only --scenarios main \
            --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done

echo "=== llama70b pilot complete ==="
