#!/bin/bash --login
#SBATCH --job-name=soo-code5-31b-ctrl
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# Reward-hacking round 5: controls for the round-2 observation that +v at
# mid alpha (4, 8) on the 31B raises ImpossibleBench test modification
# (base 1/40 -> 5/40, 6/40; whole-file reproductions with the contradicting
# assertion dropped or edited, 2 -> 8 -> 16 records per cell) while call-outs
# fall. Random norm-matched vectors at the same alphas test whether any
# mid-size perturbation does this; -v at alpha 8 tests direction.
# $1 = tag (rand_s0_a8 | rand_s1_a8 | rand_s0_a4 | steerneg_a8), $2 = alpha,
# $3 = random seed (omit for the SOO direction). Conflicting split only:
# its passed_original column already gives capability on the same tasks.
tag=${1:?tag}; alpha=${2:?alpha}; rseed=$3
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
OUT=results/code_eval/gemma4_31b
N=${N:-40}
extra=""
[ -n "$rseed" ] && extra="--steer-random-seed $rseed"
python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --data data/eval_code --scenarios impossible_conflicting \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag "$tag" \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha "$alpha" $extra
echo "=== code r5 31b $tag complete ==="
