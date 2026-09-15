#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-pilot
#SBATCH --time=10:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Qwen3.x templates default to thinking; pre-close the thought channel everywhere
export SOO_CHAT_KWARGS='{"enable_thinking": false}'

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
# Qwen3.8-27B pilot (chained on extraction). No LoRA band yet, so sweep depth
# 25-75% of 64 layers at both layer kinds: full-attention layers 15/23/31/39/47
# and Gated DeltaNet layers 16/24/32/40/48, at the alphas where gemma-4-31B
# first moved (a8) and saturated (a32); a4/a16/a64 at mid-depth bracket the
# scale in case this model's vector/activation norm ratio differs. Baselines
# in both suffix conventions as for gemma-4.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/qwen38_27b
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}
run base_room_only --suffix room_only --scenarios $SCENARIOS
run base_i_would --suffix i_would --scenarios main
for L in 15 16 23 24 31 32 39 40 47 48; do
    for A in 8 32; do
        run "steer_L${L}_a${A}" --suffix room_only --scenarios main \
            --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done
for L in 31 32; do
    for A in 4 16 64; do
        run "steer_L${L}_a${A}" --suffix room_only --scenarios main \
            --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done
echo "=== qwen38 pilot complete ==="
