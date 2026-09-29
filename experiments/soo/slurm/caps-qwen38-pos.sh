#!/bin/bash --login
#SBATCH --job-name=soo-caps-qwen38-pos
#SBATCH --time=12:00:00
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
# Round 7 part 2: capabilities for the positional cells (L31 a10, prompt-only
# and response-only). caps_steered.py applies lm_eval's chat template when
# --positions != all (the marker needs the template), so the comparable
# baseline is a chat-template alpha=0 row, run here as well.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/capabilities
python scripts/caps_steered.py $MODEL $VEC 31 0  $OUT/steer_qwen38_27b_chat_base.json --positions prompt
python scripts/caps_steered.py $MODEL $VEC 31 10 $OUT/steer_qwen38_27b_L31_a10_prompt.json --positions prompt
python scripts/caps_steered.py $MODEL $VEC 31 10 $OUT/steer_qwen38_27b_L31_a10_response.json --positions response
echo "=== qwen38 positional capabilities complete ==="
