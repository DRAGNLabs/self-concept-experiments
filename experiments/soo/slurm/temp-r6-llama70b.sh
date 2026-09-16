#!/bin/bash --login
#SBATCH --job-name=soo-temp-llama70b
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Three-GPU variant of the usual re-pin: 70B bf16 (~140 GB) shards across
# three GPUs via device_map=auto (a pair leaves <15 GB headroom). Keep
# Slurm's set if all look free; otherwise take the three emptiest.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -3 | cut -d, -f1 | paste -sd,)
    echo "Assigned set busy or unset; re-pinned to three emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Round 6 (temperature), Llama-2-70b leg — the strongest OOD-transfer model.
# Same design as the gemma legs (see temp-r6-12b.sh) minus LoRA (no 70B
# adapters; train.py is single-device). Canonical cell L16 a8.
MODEL=meta-llama/Llama-2-70b-chat-hf
VEC=results/steering/llama2_70b.pt

indist() {  # indist <tag> [eval args...]
    local tag=$1; shift
    echo "=== indist $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --scenarios main --n 50 --suffix room_only \
        --out results/steering_eval/llama2_70b --tag "$tag" "$@"
}

apollo() {  # apollo <tag> [eval args...]
    local tag=$1; shift
    echo "=== apollo $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --data data/eval_apollo --scenarios roleplaying insider_trading sandbagging_slice \
        --n 100 --suffix none --max-new-tokens 600 \
        --out results/apollo_eval/llama2_70b --tag "$tag" "$@"
}

both() {  # both <tag> [eval args...] -- uses globals T (temperature) and S (seed)
    local tag=$1; shift
    indist "$tag" --temperature "$T" --sample-seed "$S" "$@"
    apollo "ap_$tag" --temperature "$T" --sample-seed "$S" "$@"
}

T=0.7 S=0
both t07_base
both t07_steer --steer-vectors $VEC --steer-layer 16 --steer-alpha 8
both t07_rand_s0 --steer-vectors $VEC --steer-layer 16 --steer-alpha 8 --steer-random-seed 0
T=1.0
both t10_base

echo "=== temp r6 llama70b complete ==="
