#!/bin/bash --login
#SBATCH --job-name=soo-lora-g4-31b-r2
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

source ../../.venv/bin/activate

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
# Round 2 on the 31B: round 1 (13563629) found no honest band at the
# depth-matched layers (L18 inert, L26 main 0 / TH 94, L30 main 50 / TH 16).
# The 12B's band was sharp (L19=8 -> L24=92), so probe neighbors of 50%
# depth before calling the cell "none": L28/L32/L34, seed variance at L30,
# and mirrored orientation for the existing L26/L30 checkpoints.
MODEL=google/gemma-4-31B-it
SCENS="main treasure_hunt perspectives"

echo "=== baseline mirrored ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --suffix room_only --tag baseline_lora_g4_31b_mir

for L in 26 30; do
    echo "=== L${L} seed0 mirrored (round-1 checkpoint) ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
        --adapter "results/checkpoints/gemma4-31b-L${L}/seed0" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L${L}_seed0_mir"
done

for L in 28 32 34; do
    echo "=== gemma-4-31B LoRA L${L} seed 0 ==="
    python -m selfconcept.soo.train --config configs/gemma4-31b.yaml --layer ${L} --seeds 0
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma4-31b-L${L}/seed0" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L${L}_seed0"
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
        --adapter "results/checkpoints/gemma4-31b-L${L}/seed0" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L${L}_seed0_mir"
done

for seed in 1 2; do
    echo "=== gemma-4-31B LoRA L30 seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/gemma4-31b.yaml --layer 30 --seeds ${seed}
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma4-31b-L30/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L30_seed${seed}"
done

echo "=== gemma-4-31B round 2 complete ==="
