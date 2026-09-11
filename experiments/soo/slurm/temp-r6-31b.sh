#!/bin/bash --login
#SBATCH --job-name=soo-temp-31b
#SBATCH --time=30:00:00
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
# Round 6 (temperature): re-run a slice of the in-dist matrix plus all three
# Apollo operationalizations with seeded sampling (T=0.7 across conditions,
# plus a T=1.0 baseline) to test whether the greedy-decoding results are
# decoding artifacts. Slices keep compute low: in-dist main n=50;
# roleplaying / insider trading first 100; sandbagging_slice = stratified
# 50 WMDP + 50 MMLU. Per-example seeding (sample_seed x example_id) makes
# every run reproducible and subset-independent.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt

indist() {  # indist <tag> [eval args...]
    local tag=$1; shift
    echo "=== indist $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --scenarios main --n 50 --suffix room_only \
        --out results/steering_eval/gemma4_31b --tag "$tag" "$@"
}

apollo() {  # apollo <tag> [eval args...]
    local tag=$1; shift
    echo "=== apollo $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios roleplaying insider_trading sandbagging_slice \
        --n 100 --suffix none --max-new-tokens 600 \
        --out results/apollo_eval/gemma4_31b --tag "$tag" "$@"
}

both() {  # both <tag> [eval args...] -- uses globals T (temperature) and S (seed)
    local tag=$1; shift
    indist "$tag" --temperature "$T" --sample-seed "$S" "$@"
    apollo "ap_$tag" --temperature "$T" --sample-seed "$S" "$@"
}

T=0.7 S=0
both t07_base
both t07_steer --steer-vectors $VEC --steer-layer 30 --steer-alpha 16
both t07_rand_s0 --steer-vectors $VEC --steer-layer 30 --steer-alpha 16 --steer-random-seed 0
both t07_lora --adapter results/checkpoints/gemma4-31b-L32/seed0
T=1.0
both t10_base

echo "=== temp r6 31b complete ==="
