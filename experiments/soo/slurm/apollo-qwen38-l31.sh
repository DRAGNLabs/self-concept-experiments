#!/bin/bash --login
#SBATCH --job-name=soo-apollo-qwen38-l31
#SBATCH --time=08:00:00
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
# Out-of-distribution test on Apollo's roleplaying deception scenarios
# (Goldowsky-Dill et al. 2025, arXiv:2502.03407): does the SOO
set -e
# Apollo roleplaying OOD on Qwen3.8-27B, same protocol as the gemma-4 runs
# (all 371 scenarios, answer_prefix prefilled, graded afterwards by
# scripts/judge_apollo.py). Cells: base, +v at the pilot's headline cell
# L23 a8 (hardening 13716542 runs in parallel; if the mirrored random
# control fails there, rerun at L31 a16), -v, random s0, LoRA L32 seed 0
# (validated 100/100 both orientations x 3 seeds). Thinking disabled.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/apollo_eval/qwen38_27b
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios roleplaying --suffix none \
        --max-new-tokens 256 --out "$OUT" --tag "$tag" "$@"
}
# Certified cell L31 a10 (sweep 13719042); base and LoRA cells come from
# 13716550, tags carry the _L31 suffix.
run ap_steer_L31 --steer-vectors $VEC --steer-layer 31 --steer-alpha 10
run ap_neg_L31 --steer-vectors $VEC --steer-layer 31 --steer-alpha -10
run ap_rand_s0_L31 --steer-vectors $VEC --steer-layer 31 --steer-alpha 10 --steer-random-seed 0
echo "=== apollo qwen38 L31 complete ==="
