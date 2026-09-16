#!/bin/bash --login
#SBATCH --job-name=soo-steer-kimi-dev-72b
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

# Multi-GPU variant of the usual re-pin: 72B bf16 (~145 GB) shards across
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
# 2025-large matrix cell (2025-06, 72B dense, Qwen2.5-72B-based): extraction
# + pilot, same protocol as llama70b/qwen25-72b. Same 80-layer stack as its
# Qwen2.5 base -> same depth-matched grid L16/L24/L32/L40 at the standard
# pilot strengths. Pairs with Qwen2.5-72B: same architecture ~9 months of
# extra training apart, isolating the age axis at constant size.
MODEL=moonshotai/Kimi-Dev-72B
VEC=results/steering/kimi_dev_72b.pt
OUT=results/steering_eval/kimi_dev_72b

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

echo "=== kimi-dev-72b pilot complete ==="
