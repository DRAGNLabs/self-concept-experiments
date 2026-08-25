#!/bin/bash --login
#SBATCH --job-name=soo-audit-fixes
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/jansen05/self-concept-experiments

mamba activate ./.env

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
MISTRAL=mistralai/Mistral-7B-Instruct-v0.2
OLMO=allenai/OLMo-2-1124-7B-Instruct
SCENS="main treasure_hunt perspectives"

# Codex-audit follow-ups:
# 1. Latent SOO (attn+mlp sites, train/test probe splits) for the L16
#    checkpoints — the selected reproduction had no latent measurement.
# 2. Positional-confound test: re-eval on the mirrored eval sets (honest room
#    mentioned first; identical fills). If SOO checkpoints learned a position
#    heuristic instead of honesty, mirrored deception rates will invert.

echo "=== latent SOO: baseline + L16 seeds ==="
python -m soo.latent_soo --model "$MISTRAL" --tag baseline_mistral7b_v2
for seed in 0 1 2 3 4; do
    python -m soo.latent_soo --model "$MISTRAL" \
        --adapter "results/checkpoints/mistral-7b-lasttok-L16/seed${seed}" \
        --tag "soo_mistral7b_lasttok_L16_seed${seed}"
done

echo "=== mirrored eval: Mistral baseline n=250 ==="
python -m soo.evaluate --model "$MISTRAL" --data data/eval_mirrored \
    --scenarios $SCENS --n 250 --tag baseline_mistral7b_mir_n250

for seed in 0 1 2 3 4; do
    echo "=== mirrored eval: L16 seed ${seed} n=250 ==="
    python -m soo.evaluate --model "$MISTRAL" --data data/eval_mirrored \
        --adapter "results/checkpoints/mistral-7b-lasttok-L16/seed${seed}" \
        --scenarios $SCENS --n 250 --tag "soo_mistral7b_lasttok_L16_seed${seed}_mir_n250"
done

echo "=== mirrored eval: OLMo-7B baseline + seeds 0-1 (n=50) ==="
python -m soo.evaluate --model "$OLMO" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --tag baseline_olmo7b_mir
for seed in 0 1; do
    python -m soo.evaluate --model "$OLMO" --data data/eval_mirrored \
        --adapter "results/checkpoints/olmo2-7b/seed${seed}" \
        --scenarios $SCENS --n 50 --tag "soo_olmo7b_seed${seed}_mir"
done

echo "=== audit fixes complete ==="
