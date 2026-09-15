#!/bin/bash --login
#SBATCH --job-name=soo-steer-extract-qwen38
#SBATCH --time=03:00:00
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
# Qwen3.8-27B (Qwen, 2026-08): 2026 ~30B-class model from a third lineage next
# to gemma-4-31B and Muse-30B in the matrix. Hybrid stack (48 Gated DeltaNet +
# 16 full-attention layers): vectors are taken at each layer's attention
# output projection (o_proj / out_proj, see activations.attn_out_proj).
# Thinking off via SOO_CHAT_KWARGS so prompts end in an empty, closed think
# block like gemma-4's default. Ends with a 5-example base generation so the
# log shows whether the model answers in the direct-answer format at all.
MODEL=Qwen/Qwen3.8-27B
python scripts/extract_steering.py "$MODEL" results/steering/qwen38_27b.pt
python -m selfconcept.soo.evaluate --model "$MODEL" --n 5 --suffix room_only --scenarios main \
    --out results/steering_eval/qwen38_27b --tag smoke_base
echo "=== qwen38 extraction complete ==="
