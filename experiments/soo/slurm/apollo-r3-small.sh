#!/bin/bash --login
#SBATCH --job-name=soo-apollo3-small
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
# Apollo OOD round 3, small/negative-control tier: the taxonomy predicts
# NO transfer here — Mistral inert, OLMo harmed (anti-honest), gemma-2
# partial in-distribution. 4 conditions x 371 each at the canonical
# in-distribution cells.
apollo() {  # apollo <out_key> <model> <extra evaluate args...>
    local key=$1 model=$2; shift 2
    python -m selfconcept.soo.evaluate --model "$model" \
        --data data/eval_apollo --scenarios roleplaying --suffix none \
        --max-new-tokens 256 --out "results/apollo_eval/$key" "$@"
}
for spec in "mistral7b mistralai/Mistral-7B-Instruct-v0.2 mistral7b.pt 16 8" \
            "olmo7b allenai/OLMo-2-1124-7B-Instruct olmo7b.pt 19 16" \
            "gemma27b google/gemma-2-27b-it gemma27b.pt 14 32"; do
    set -- $spec; key=$1 model=$2 vec=results/steering/$3 L=$4 A=$5
    echo "=== $key ap_base ==="
    apollo "$key" "$model" --tag ap_base
    echo "=== $key ap_steer ==="
    apollo "$key" "$model" --tag ap_steer --steer-vectors $vec --steer-layer $L --steer-alpha $A
    echo "=== $key ap_neg ==="
    apollo "$key" "$model" --tag ap_neg --steer-vectors $vec --steer-layer $L --steer-alpha -$A
    echo "=== $key ap_rand_s0 ==="
    apollo "$key" "$model" --tag ap_rand_s0 --steer-vectors $vec --steer-layer $L --steer-alpha $A --steer-random-seed 0
done
echo "=== apollo round3 small complete ==="
