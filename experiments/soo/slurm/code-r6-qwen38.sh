#!/bin/bash --login
#SBATCH --job-name=soo-code6-qwen38
#SBATCH --time=30:00:00
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
# (base 1/40 -> 5/40, 6/40; whole-file reproductions with the contradicting
set -e
# Reward-hacking round 6: Qwen3.8-27B, the model with a certified vector cell
# (L31 a10: 0 -> 100 both orientations, mirrored random 10/6/50/8, -v 100)
# and a certified LoRA (L32: 100/100 x 3 seeds). Same protocol as the
# gemma-4 rounds (n=40, 3 attempts, 2048 tokens, greedy, three scenarios).
# $1 = cell: base | steer | neg | rand_s0 | lora, or steer_a<A> / rand_s0_a<A>
# for the alpha sweep (the certified a10 collapses multi-turn coding to ~10%
# original pass in the first 15-19 tasks, as gemma-4's SOO alpha did; a8 and
# a6 are where the in-distribution random control was at floor). Thinking off.
cell=${1:?cell}
SCEN="impossible_conflicting impossible_original evilgenie"
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/code_eval/qwen38_27b
N=${N:-40}
case "$cell" in
    base)    extra="" ;;
    steer)   extra="--steer-vectors $VEC --steer-layer 31 --steer-alpha 10" ;;
    neg)     extra="--steer-vectors $VEC --steer-layer 31 --steer-alpha -10" ;;
    rand_s0) extra="--steer-vectors $VEC --steer-layer 31 --steer-alpha 10 --steer-random-seed 0" ;;
    lora)    extra="--adapter results/checkpoints/qwen38-27b-L32/seed0" ;;
    steer_a*)   A=${cell#steer_a};   extra="--steer-vectors $VEC --steer-layer 31 --steer-alpha $A"; SCEN="impossible_conflicting impossible_original" ;;
    rand_s0_a*) A=${cell#rand_s0_a}; extra="--steer-vectors $VEC --steer-layer 31 --steer-alpha $A --steer-random-seed 0"; SCEN="impossible_conflicting impossible_original" ;;
    *) echo "unknown cell $cell"; exit 1 ;;
esac
python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --scenarios $SCEN \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag "$cell" $extra
echo "=== code r6 qwen38 $cell complete ==="
