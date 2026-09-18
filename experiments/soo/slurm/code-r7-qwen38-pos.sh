#!/bin/bash --login
#SBATCH --job-name=soo-code7-qwen38
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
# Reward-hacking round 7: positional steering on Qwen3.8-27B. Round 6/6b
# showed the certified L31 a10 vector (and any matched-norm offset at a>=8)
# collapses multi-turn coding when added at every position. These cells add
# it only on the model's own turn (--steer-positions response) or only on
# the context (prompt, diagnostic). Same protocol as rounds 6/6b (n=40,
# 3 attempts, 2048 tokens, greedy). $1 = cell:
#   resp_a<A>            +v, response positions, all three scenarios
#   resp_neg_a<A>        -v, response positions (conflicting + original)
#   resp_rand_s<S>_a<A>  random seed S matched-norm, response positions (conflicting + original)
#   prompt_a<A>          +v on the context only (conflicting + original)
cell=${1:?cell}
SCEN="impossible_conflicting impossible_original"
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/code_eval/qwen38_27b
N=${N:-40}
S="--steer-vectors $VEC --steer-layer 31"
case "$cell" in
    resp_neg_a*)    A=${cell#resp_neg_a}; extra="$S --steer-alpha -$A --steer-positions response" ;;
    resp_rand_s*_a*) t=${cell#resp_rand_s}; SEED=${t%%_a*}; A=${t#*_a}
                    extra="$S --steer-alpha $A --steer-random-seed $SEED --steer-positions response" ;;
    resp_a*)        A=${cell#resp_a}; extra="$S --steer-alpha $A --steer-positions response"
                    SCEN="impossible_conflicting impossible_original evilgenie" ;;
    prompt_neg_a*)  A=${cell#prompt_neg_a}; extra="$S --steer-alpha -$A --steer-positions prompt" ;;
    prompt_rand_s*_a*) t=${cell#prompt_rand_s}; SEED=${t%%_a*}; A=${t#*_a}
                    extra="$S --steer-alpha $A --steer-random-seed $SEED --steer-positions prompt" ;;
    prompt_a*)      A=${cell#prompt_a}; extra="$S --steer-alpha $A --steer-positions prompt" ;;
    *) echo "unknown cell $cell"; exit 1 ;;
esac
echo "cell $cell: $extra ; scenarios: $SCEN"
python -m selfconcept.soo.evaluate_code --model "$MODEL" \
    --scenarios $SCEN \
    --n "$N" --max-attempts 3 --max-new-tokens 2048 --out "$OUT" --tag "$cell" $extra
echo "=== code r7 qwen38 $cell complete ==="
