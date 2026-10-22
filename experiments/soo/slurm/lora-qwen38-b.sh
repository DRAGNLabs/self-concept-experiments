#!/bin/bash --login
#SBATCH --job-name=soo-lora-qwen38-b
#SBATCH --time=16:00:00
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
# Qwen3.8-27B LoRA layer sweep, part b (chained on extraction): Gemma-2
# paper recipe (configs/qwen38-27b.yaml), one seed per loss layer. Layers
# bracket the gemma-4 LoRA band (~50% depth, razor-sharp on the 31B) with
# both layer kinds: L19 (30%, full attn), L27 (42%, full), L31 (48%, full),
# L32 (50%, DeltaNet), L35 (55%, full). Seeds 1-2 follow at whichever holds.
MODEL=Qwen/Qwen3.8-27B
SCENS="main treasure_hunt perspectives"
OUT=results/steering_eval/qwen38_27b
for L in 27 35; do
    echo "=== Qwen3.8-27B LoRA L${L} seed 0 ==="
    python -m selfconcept.soo.train --config configs/qwen38-27b.yaml --layer ${L} --seeds 0
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/qwen38-27b-L${L}/seed0" \
        --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag "lora_L${L}_seed0"
done
echo "=== qwen38 lora sweep part b complete ==="
