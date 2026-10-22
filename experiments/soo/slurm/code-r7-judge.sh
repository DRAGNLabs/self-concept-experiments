#!/bin/bash --login
#SBATCH --job-name=soo-code7-judge
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
# Round-7 judge: positional-steering cells' conflicting + EvilGenie splits,
# graded by the same Qwen2.5-72B as every other code cell; then the
# messages-only call-out pass. $@ = cell tags.
[ $# -gt 0 ] || { echo "usage: $0 <tag>..."; exit 1; }
files=""
for tag in "$@"; do
    for scen in impossible_conflicting evilgenie; do
        f=results/code_eval/qwen38_27b/${tag}_${scen}.jsonl
        [ -f "$f" ] && files="$files $f"
    done
done
[ -n "$files" ] || { echo "no result files"; exit 1; }
echo "judging: $files"
python -m selfconcept.codebench.judge --model Qwen/Qwen2.5-72B-Instruct --responses $files --batch-size 4 --max-new-tokens 512 --skip-existing
python -m selfconcept.codebench.callout --model Qwen/Qwen2.5-72B-Instruct --responses $(for f in $files; do case $f in *conflicting*) echo -n "$f ";; esac; done) --rejudge-errors
echo "=== code r7 judge complete ==="
