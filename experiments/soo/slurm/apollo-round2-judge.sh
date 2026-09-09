#!/bin/bash --login
#SBATCH --job-name=soo-apollo2-judge
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Three-GPU variant of the usual re-pin: 72B bf16 (~145 GB) shards across
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
# Grade only the round-2 files (round-1 grades are final; regrading
# them would waste ~5 GPU-hours for identical output). Submitted with
# --dependency=afterok on the two round-2 generation jobs.
python scripts/judge_apollo.py \
    --responses results/apollo_eval/gemma4_31b/ap_steer_a8_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_steer_a24_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_rand_s1_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_lora_s1_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_lora_s2_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_lora_s1_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_lora_s2_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_rand_s1_roleplaying_none.jsonl \
    --batch-size 16 --max-new-tokens 256

echo "=== apollo round2 judge complete ==="
