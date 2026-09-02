#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma4-12b
#SBATCH --time=08:00:00
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
# gemma-4-12B-it: the newer-smaller cell of the age x size matrix — same
# family and generation as the 31B, pure size axis. 48 layers, hidden 3840
# (Gemma4UnifiedForConditionalGeneration; loads via AutoModelForCausalLM).
# Extraction then the standard pilot: depth sweep at 30/40/50/60% depth
# (L14/L19/L24/L29 of 48) x alpha {8,32}, both-suffix baselines. A random
# control at the 50%-depth cell is included at pilot stage this time — the
# 31B's hardening showed matched-norm random flips it too, so check early.
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma4_12b

python scripts/extract_steering.py $MODEL $VEC

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

run base_room_only --suffix room_only --scenarios $SCENARIOS
run base_i_would --suffix i_would --scenarios main
for L in 14 19 24 29; do
    for A in 8 32; do
        run "steer_L${L}_a${A}" --suffix room_only --scenarios main \
            --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done
run steer_L24_rand_a32 --suffix room_only --scenarios main \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32 --steer-random-seed 0

echo "=== gemma4-12b extract+pilot complete ==="
