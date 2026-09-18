#!/bin/bash --login
#SBATCH --job-name=soo-lora-qwen38-agentic
#SBATCH --time=18:00:00
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
# Qwen3.x templates default to thinking; pre-close the thought channel everywhere
export SOO_CHAT_KWARGS='{"enable_thinking": false}'

# GPU isolation isn't enforced on these nodes and Slurm's CUDA_VISIBLE_DEVICES
# can point at a GPU already occupied by other/orphaned processes (seen on
# dw-2-4) — or at a GPU another job holds exclusively. Keep the assigned GPU
# if it looks free; otherwise re-pin to the emptiest visible one.
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
# Round 8: agentic SOO data (selfconcept.soo.agentic). The original adapters
# transfer to insider trading but not to Apollo roleplaying on gemma-4-31B
# and Qwen3.8-27B; this trains the unchanged L32 recipe on self/other pairs
# written in the roleplaying / agentic-report style (60 pairs, 20 situations
# x 3 templates) or on those plus the 78 original pairs (mixed), then runs
# the in-distribution scenarios (does the toy task still flip?), Apollo
# roleplaying (the target gap) and insider trading (must not regress).
# $1 = agentic | mixed. Judges chained separately (apollo-qwen38-agentic-judge.sh).
variant=${1:?agentic|mixed|agentic-noinsider|agentic-persons}
# SEEDS (env, default "0 1 2") and STAGES (env, default "indist roleplaying
# insider") narrow a run, e.g. the round-8b single-seed diagnostics.
SEEDS=${SEEDS:-0 1 2}; STAGES=${STAGES:-indist roleplaying insider}
MODEL=Qwen/Qwen3.8-27B
SCENS="main treasure_hunt perspectives"
OUT=results/steering_eval/qwen38_27b
APOLLO=results/apollo_eval/qwen38_27b
CK=results/checkpoints/qwen38-27b-${variant}-L32
for seed in $SEEDS; do
    if [ ! -f "$CK/seed${seed}/adapter_config.json" ]; then
        echo "=== train $variant L32 seed $seed ==="
        python -m selfconcept.soo.train --config configs/qwen38-27b-${variant}.yaml --layer 32 --seeds $seed
    fi
done
[[ " $STAGES " == *" indist "* ]] && for seed in $SEEDS; do
    A="--adapter $CK/seed${seed}"
    echo "=== in-distribution $variant seed $seed ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" $A --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag "lora_${variant}_L32_seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" $A --data data/eval_mirrored --scenarios $SCENS --n 50 --suffix room_only --out "$OUT" --tag "lora_${variant}_L32_seed${seed}_mir"
done
[[ " $STAGES " == *" roleplaying "* ]] && for seed in $SEEDS; do
    A="--adapter $CK/seed${seed}"
    echo "=== roleplaying $variant seed $seed ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" $A --data data/eval_apollo --scenarios roleplaying --suffix none --max-new-tokens 256 --out "$APOLLO" --tag "ap_lora_${variant}_s${seed}"
done
[[ " $STAGES " == *" insider "* ]] && for seed in $SEEDS; do
    A="--adapter $CK/seed${seed}"
    echo "=== insider $variant seed $seed ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" $A --data data/eval_apollo --scenarios insider_trading --suffix none --max-new-tokens 600 --out "$APOLLO" --tag "ap_lora_${variant}_s${seed}"
done
echo "=== lora qwen38 $variant complete ==="
