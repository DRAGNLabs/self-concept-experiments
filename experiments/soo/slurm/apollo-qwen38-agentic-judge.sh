#!/bin/bash --login
#SBATCH --job-name=soo-apollo-qwen38-agentic-judge
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
# Round-8 judges: the agentic / mixed adapters' roleplaying and insider
# reports (3 seeds each), same Qwen2.5-72B judges as every other Apollo cell.
D=results/apollo_eval/qwen38_27b
rp=$(ls $D/ap_lora_{agentic,mixed}_s{0,1,2}_roleplaying_none.jsonl 2>/dev/null || true)
it=$(ls $D/ap_lora_{agentic,mixed}_s{0,1,2}_insider_trading_none.jsonl 2>/dev/null || true)
echo "roleplaying: $rp"; echo "insider: $it"
[ -n "$rp" ] && python scripts/judge_apollo.py --responses $rp --batch-size 16 --max-new-tokens 256
[ -n "$it" ] && python scripts/judge_insider.py --responses $it --batch-size 16 --max-new-tokens 256
echo "=== apollo qwen38 agentic judge complete ==="
