#!/bin/bash --login
#SBATCH --job-name=soo-apollo5-12b
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

source ../../.venv/bin/activate

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
# Apollo OOD round 5: seed certification of the round-4 reversals on the two
# new operationalizations: LoRA s1-s2 (12B insider backfire, dec 50% s0) and
# rand s1 (s0 broke format, 54% invalid insider). Same 600-token budget
# as round 4 for comparability.
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
OUT=results/apollo_eval/gemma4_12b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios insider_trading sandbagging \
        --suffix none --max-new-tokens 600 --out "$OUT" --tag "$tag" "$@"
}


run ap_lora_s1 --adapter results/checkpoints/gemma4-12b-L24/seed1
run ap_lora_s2 --adapter results/checkpoints/gemma4-12b-L24/seed2
run ap_rand_s1 --steer-vectors $VEC --steer-layer 19 --steer-alpha 12 --steer-random-seed 1

echo "=== apollo r5 12b complete ==="
