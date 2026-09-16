#!/bin/bash --login
#SBATCH --job-name=soo-code1-judge
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
# Judge every round-1 coding result file that exists (originally submitted
# --dependency=afterok on all eight generation jobs). ImpossibleBench original
# needs no judge (pass = solved). Optional $1 = 12b|31b restricts to one model
# size (the rerun after job 13692242 OOMed at batch 8 runs the two sizes as
# separate jobs; --skip-existing keeps the file it had finished).
sizes=${1:-"12b 31b"}
files=$(for sz in $sizes; do ls results/code_eval/gemma4_$sz/{base,steer,rand_s0,lora_s0}_{impossible_conflicting,evilgenie}.jsonl 2>/dev/null; done || true)
[ -n "$files" ] || { echo "no result files for sizes '$sizes'"; exit 1; }
echo "judging: $files"
python -m selfconcept.codebench.judge --responses $files --batch-size 4 --max-new-tokens 512 --skip-existing

echo "=== code r1 judge complete ==="
