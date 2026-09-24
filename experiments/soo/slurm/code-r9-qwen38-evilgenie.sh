#!/bin/bash --login
#SBATCH --job-name=soo-code9-eg-qwen38
#SBATCH --time=14:00:00
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
export SOO_CHAT_KWARGS='{"enable_thinking": false}'
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
# Reward-hacking round 9 (2026-09-24): re-run of every Qwen3.8-27B EvilGenie
# cell that feeds a claim (audit paired table, round 8 code table, round 7
# response-only cell) on the fixed harness. Until 2026-09-24 the workspace
# files shown to the model went through the 3000-char middle truncation, so
# test.py was cut on every problem and most visible tests were hidden
# (FINDINGS.md "EvilGenie harness defect"). Same protocol as round 6 (n=40,
# 3 attempts, 2048 tokens, greedy, thinking off), EvilGenie only (the
# ImpossibleBench prompts were never truncated), new output directory so the
# old records stay for comparison. $1 = cell:
#   base | lora | lora_agentic | lora_mixed | steer | neg | rand_s0 | resp_a10
cell=${1:?cell}
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/code_eval/qwen38_27b_eg_fixed
N=${N:-40}
S="--steer-vectors $VEC --steer-layer 31"
case "$cell" in
    base)         extra="" ;;
    lora)         extra="--adapter results/checkpoints/qwen38-27b-L32/seed0" ;;
    lora_agentic) extra="--adapter results/checkpoints/qwen38-27b-agentic-L32/seed0" ;;
    lora_mixed)   extra="--adapter results/checkpoints/qwen38-27b-mixed-L32/seed0" ;;
    steer)        extra="$S --steer-alpha 10" ;;
    neg)          extra="$S --steer-alpha -10" ;;
    rand_s0)      extra="$S --steer-alpha 10 --steer-random-seed 0" ;;
    resp_a10)     extra="$S --steer-alpha 10 --steer-positions response" ;;
    *) echo "unknown cell $cell"; exit 1 ;;
esac
echo "harness commit: $(git -C ../.. rev-parse HEAD)"
echo "cell $cell: $extra"
python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --scenarios evilgenie \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag "$cell" $extra
echo "=== code r9 evilgenie qwen38 $cell complete ==="
