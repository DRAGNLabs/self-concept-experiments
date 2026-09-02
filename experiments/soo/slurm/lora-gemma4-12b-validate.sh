#!/bin/bash --login
#SBATCH --job-name=soo-lora-g4-12b-val
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
# Validate the 12B sweep winner (L24: main 0->92, TH 0->100, persp 92):
# mirrored orientation (the OLMo-style positional-confound test — TH 0->100
# at every swept layer is exactly that signature), two more seeds, n=250.
MODEL=google/gemma-4-12B-it
SCENS="main treasure_hunt perspectives"

echo "=== baseline mirrored ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --suffix room_only --tag baseline_lora_g4_12b_mir

echo "=== seed0 mirrored ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --adapter results/checkpoints/gemma4-12b-L24/seed0 \
    --scenarios $SCENS --n 50 --suffix room_only --tag soo_g4_12b_L24_seed0_mir

echo "=== seed0 n250 main both orientations ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --adapter results/checkpoints/gemma4-12b-L24/seed0 \
    --scenarios main --n 250 --suffix room_only --tag soo_g4_12b_L24_seed0_n250
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --adapter results/checkpoints/gemma4-12b-L24/seed0 \
    --scenarios main --n 250 --suffix room_only --tag soo_g4_12b_L24_seed0_mir_n250

for seed in 1 2; do
    echo "=== gemma-4-12B LoRA L24 seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/gemma4-12b.yaml --layer 24 --seeds ${seed}
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma4-12b-L24/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_12b_L24_seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
        --adapter "results/checkpoints/gemma4-12b-L24/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_12b_L24_seed${seed}_mir"
done

echo "=== gemma-4-12B L24 validation complete ==="
