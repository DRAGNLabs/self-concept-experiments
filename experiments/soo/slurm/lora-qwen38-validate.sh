#!/bin/bash --login
#SBATCH --job-name=soo-lora-qwen38-val
#SBATCH --time=05:00:00
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
# Qwen3.8-27B LoRA L32 validation (sweep part a, job 13705500: L32 seed 0
# main 0 -> 100, TH -> 100, persp 100 with 11-12 distinct room answers per
# scenario; L31 48/38, L19 22/0). Same treatment gemma-4's L24/L32 cells
# got before their verdicts: base rates at n=50 in both orientations (the
# only base numbers so far are a 5-example smoke), seed0 mirrored, seed0
# n=250 both orientations, seeds 1-2 both orientations. L33 seed 0 (the
# next DeltaNet layer) probes the band width beyond part b's L27/L35.
MODEL=Qwen/Qwen3.8-27B
SCENS="main treasure_hunt perspectives"
OUT=results/steering_eval/qwen38_27b

echo "=== baseline orig ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag base
echo "=== baseline mirrored ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag base_mir

echo "=== L32 seed0 mirrored ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --adapter results/checkpoints/qwen38-27b-L32/seed0 \
    --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag lora_L32_seed0_mir

echo "=== L32 seed0 n250 main both orientations ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --adapter results/checkpoints/qwen38-27b-L32/seed0 \
    --scenarios main --n 250 --suffix room_only --out "$OUT" --tag lora_L32_seed0_n250
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --adapter results/checkpoints/qwen38-27b-L32/seed0 \
    --scenarios main --n 250 --suffix room_only --out "$OUT" --tag lora_L32_seed0_mir_n250

for seed in 1 2; do
    echo "=== Qwen3.8-27B LoRA L32 seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/qwen38-27b.yaml --layer 32 --seeds ${seed}
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/qwen38-27b-L32/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag "lora_L32_seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
        --adapter "results/checkpoints/qwen38-27b-L32/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag "lora_L32_seed${seed}_mir"
done

echo "=== Qwen3.8-27B LoRA L33 seed 0 (band width) ==="
python -m selfconcept.soo.train --config configs/qwen38-27b.yaml --layer 33 --seeds 0
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --adapter results/checkpoints/qwen38-27b-L33/seed0 \
    --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag lora_L33_seed0

echo "=== qwen38 lora L32 validation complete ==="
