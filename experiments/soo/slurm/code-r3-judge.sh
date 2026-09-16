#!/bin/bash --login
#SBATCH --job-name=soo-code3-judge
#SBATCH --time=06:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# Round-3 screen judge: every non-gemma base file that exists at submit time
# (Muse conflicting/EvilGenie, Qwen2.5-72B conflicting/EvilGenie, Kimi-Dev
# conflicting/EvilGenie); --skip-existing lets a later resubmission pick up
# the rest. Submitted --dependency=afterok on the generation jobs still
# running, because evaluate_code appends records incrementally and a
# partial file would be graded and then skipped for good.
# Qwen2.5-72B grading its own outputs is a known caveat (PLAN).
files=$(ls results/code_eval/{muse30b,qwen25_72b,kimi_dev_72b}/base_{impossible_conflicting,evilgenie}.jsonl 2>/dev/null || true)
[ -n "$files" ] || { echo "no result files"; exit 1; }
echo "judging: $files"
python scripts/judge_code.py --responses $files --batch-size 4 --max-new-tokens 512 --skip-existing
python scripts/judge_callout.py --responses $files --rejudge-errors
echo "=== code r3 judge complete ==="
