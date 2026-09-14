#!/bin/bash --login
#SBATCH --job-name=soo-code1-12b-rand
#SBATCH --time=36:00:00
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
# Reward-hacking coding round 1 (12b, condition rand): ImpossibleBench
# conflicting (cheat rate) + original (capability control) + EvilGenie, first
# N tasks each, 3 submissions per task. Same steer layer/alpha and LoRA
# seed-0 adapter as the Apollo rounds. One condition per job: the smoke test
# measured ~13 tok/s, i.e. up to ~8 min per 3-attempt task at 2048 tokens.
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
OUT=results/code_eval/gemma4_12b
N=${N:-40}

python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --data data/eval_code --scenarios impossible_conflicting impossible_original evilgenie \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag rand_s0 --steer-vectors $VEC --steer-layer 19 --steer-alpha 12 --steer-random-seed 0

echo "=== code r1 12b rand complete ==="
