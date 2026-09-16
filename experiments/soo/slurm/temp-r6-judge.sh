#!/bin/bash --login
#SBATCH --job-name=soo-temp-judge
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

source ../../.venv/bin/activate

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
# Grade the round-6 (temperature) generations: 14 roleplaying files (1-7
# honesty grades) and 14 insider trading files (explicit/implied/concealed),
# 100 examples each; judging itself stays greedy. sandbagging_slice needs no
# judge (scripts/parse_sandbagging.py on the login node). Submitted with
# --dependency=afterok on the three r6 generation jobs.

python scripts/judge_apollo.py \
    --responses results/apollo_eval/gemma4_12b/ap_t07_base_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_steer_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_rand_s0_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_lora_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t10_base_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_base_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_steer_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_rand_s0_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_lora_roleplaying_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t10_base_roleplaying_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_base_roleplaying_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_steer_roleplaying_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_rand_s0_roleplaying_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t10_base_roleplaying_none.jsonl \
    --batch-size 16 --max-new-tokens 256

python scripts/judge_insider.py \
    --responses results/apollo_eval/gemma4_12b/ap_t07_base_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_steer_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_rand_s0_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t07_lora_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_12b/ap_t10_base_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_base_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_steer_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_rand_s0_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t07_lora_insider_trading_none.jsonl \
                results/apollo_eval/gemma4_31b/ap_t10_base_insider_trading_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_base_insider_trading_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_steer_insider_trading_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t07_rand_s0_insider_trading_none.jsonl \
                results/apollo_eval/llama2_70b/ap_t10_base_insider_trading_none.jsonl \
    --batch-size 16 --max-new-tokens 256

echo "=== temp r6 judge complete ==="
