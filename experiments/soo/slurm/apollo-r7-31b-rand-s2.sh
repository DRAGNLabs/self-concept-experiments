#!/bin/bash --login
#SBATCH --job-name=soo-apollo7-31b
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
# Apollo OOD round 7: third random seed for the 31B controls. Round 5 left
# a wide two-seed rand spread on insider (honest 0.6% s0 vs 17.9% s1)
# behind the newly 3-seed-certified LoRA transfer (45/61/49%); rand s2
# tightens that margin, and the sandbagging half tightens the
# backfire-is-LoRA-specific contrast. Same 600-token budget as rounds 4-5.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
OUT=results/apollo_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios insider_trading sandbagging \
        --suffix none --max-new-tokens 600 --out "$OUT" --tag "$tag" "$@"
}


run ap_rand_s2 --steer-vectors $VEC --steer-layer 30 --steer-alpha 16 --steer-random-seed 2

echo "=== apollo r7 31b complete ==="
