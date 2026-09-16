#!/bin/bash --login
#SBATCH --job-name=soo-code3-muse-base
#SBATCH --time=36:00:00
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
# Reward-hacking coding round 3 = model screen, Muse-Glimmer-30B base: does a
# model outside the gemma-4 pair reward-hack at all on ImpossibleBench
# conflicting / EvilGenie (gemma-4 barely does, so SOO had nothing to reduce)?
# Same protocol as round 1 (n=40, 3 attempts, 2048 tokens, greedy); the
# +v/random cells (L26 a8 per STEERING_MATRIX) follow only if base hacks.
# --force-user-channel as in the Apollo runs (Muse otherwise opens a to=self
# channel). Muse has no honest LoRA regime, so it is a vector-only model.
MODEL=meta-models/Muse-Glimmer-30B
OUT=results/code_eval/muse30b
N=${N:-40}
python -m selfconcept.soo.evaluate_code --model "$MODEL" --force-user-channel \
    --scenarios impossible_conflicting impossible_original evilgenie \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag base
echo "=== code r3 muse base complete ==="
