#!/bin/bash --login
#SBATCH --job-name=soo-lora-g4-31b-r3
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
# Round 3 on the 31B: round 2 (13565579) found the band at L32 — seed0
# hits main 100/TH 100/persp 100 orig and 100/96/100 mirrored with clean
# varied responses. But L30's round-1 main 50 (seed0) collapsed to 0 in
# seeds 1-2, so this model is seed-fragile off-band; L32 gets the same
# treatment the 12B's L24 got before its verdict: seeds 1-2 in both
# orientations, and n=250 anchors on seed0.
MODEL=google/gemma-4-31B-it
SCENS="main treasure_hunt perspectives"

for seed in 1 2; do
    echo "=== gemma-4-31B LoRA L32 seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/gemma4-31b.yaml --layer 32 --seeds ${seed}
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma4-31b-L32/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L32_seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
        --adapter "results/checkpoints/gemma4-31b-L32/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L32_seed${seed}_mir"
done

echo "=== L32 seed0 n250 anchors ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --adapter "results/checkpoints/gemma4-31b-L32/seed0" \
    --scenarios main --n 250 --suffix room_only --tag "soo_g4_31b_L32_seed0_n250"
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --adapter "results/checkpoints/gemma4-31b-L32/seed0" \
    --scenarios main --n 250 --suffix room_only --tag "soo_g4_31b_L32_seed0_mir_n250"

echo "=== gemma-4-31B round 3 complete ==="
