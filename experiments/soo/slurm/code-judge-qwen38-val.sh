#!/bin/bash --login
#SBATCH --job-name=soo-code-judge-qwen38-val
#SBATCH --time=04:00:00
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

# GPU isolation isn't enforced on these nodes; keep the assigned GPU if it
# looks free, otherwise re-pin to the emptiest visible one.
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
# Reward-hacking coding round 1 (12b, condition base): ImpossibleBench
# conflicting (cheat rate) + original (capability control) + EvilGenie, first
# N tasks each, 3 submissions per task. Same steer layer/alpha and LoRA
set -e
# Judge-swap validation: the codebench default judge moved from
# Qwen2.5-72B (3 GPUs) to Qwen3.8-27B (one GPU, thinking off). Grade three
# already-72B-graded gemma-4-12B files with the new default into a separate
# --out-dir so the two judges can be compared record by record (agreement on
# categories, call-out, EvilGenie labels) before the new default is trusted.
# First attempt at 512 output tokens truncated 23/40 analyses before the
# JSON; rerun at the package default (now 1024).
export SOO_CHAT_KWARGS='{"enable_thinking": false}'
files="results/code_eval/gemma4_12b/base_impossible_conflicting.jsonl results/code_eval/gemma4_12b/lora_s0_impossible_conflicting.jsonl results/code_eval/gemma4_12b/base_evilgenie.jsonl"
python -m selfconcept.codebench.judge --responses $files --out-dir results/code_eval/judge_qwen38 --batch-size 4
python -m selfconcept.codebench.callout --responses $files --out-dir results/code_eval/judge_qwen38 --rejudge-errors
echo "=== code judge qwen38 validation complete ==="
