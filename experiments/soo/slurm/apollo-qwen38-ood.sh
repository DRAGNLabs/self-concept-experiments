#!/bin/bash --login
#SBATCH --job-name=soo-apollo-qwen38-ood
#SBATCH --time=24:00:00
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
# Out-of-distribution test on Apollo's roleplaying deception scenarios
set -e
# Insider trading + sandbagging OOD on Qwen3.8-27B, same protocol as the
# gemma-4 round-4/5 runs (600 tokens, suffix none; insider reports graded
# by scripts/judge_insider.py, sandbagging parsed on the login node by
# scripts/parse_sandbagging.py). Cells as in apollo-qwen38.sh: base, +v
# L23 a8, -v, random s0, LoRA L32 seed 0. Thinking disabled.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/apollo_eval/qwen38_27b
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios insider_trading sandbagging \
        --suffix none --max-new-tokens 600 --out "$OUT" --tag "$tag" "$@"
}
run ap_base
run ap_steer --steer-vectors $VEC --steer-layer 23 --steer-alpha 8
run ap_neg --steer-vectors $VEC --steer-layer 23 --steer-alpha -8
run ap_rand_s0 --steer-vectors $VEC --steer-layer 23 --steer-alpha 8 --steer-random-seed 0
run ap_lora --adapter results/checkpoints/qwen38-27b-L32/seed0
echo "=== apollo qwen38 ood complete ==="
