#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-l31a8
#SBATCH --time=05:00:00
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
# L31 a10's third random seed leaks (s2: 94 orig / 50 mirrored vs s0/s1
# 18/44 and 10/6), so the a10 certification is "real 100 vs random 6-50
# mirrored". At a8 seeds 0/1 were at floor in both orientations (4/2,
# 0/0) with real 80/92: complete that cell the same way (random s2, TH and
# perspectives, n=250, both orientations) and add random s3 at a10 both
# orientations to size the random spread, then pick the headline between
# a8 (cleaner control) and a10 (stronger effect).
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/steering_eval/qwen38_27b
MIR="--data data/eval_mirrored"
S8="--steer-vectors $VEC --steer-layer 31 --steer-alpha 8"
S10="--steer-vectors $VEC --steer-layer 31 --steer-alpha 10"
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --out "$OUT" --tag "$tag" "$@"
}
run steer_L31_rand2_a8 --n 50 --scenarios main $S8 --steer-random-seed 2
run steer_L31_rand2_a8_mir --n 50 $MIR --scenarios main $S8 --steer-random-seed 2
run steer_L31_rand3_a10 --n 50 --scenarios main $S10 --steer-random-seed 3
run steer_L31_rand3_a10_mir --n 50 $MIR --scenarios main $S10 --steer-random-seed 3
run steer_L31_a8 --n 50 --scenarios perspectives treasure_hunt $S8
run steer_L31_a8_mir --n 50 $MIR --scenarios perspectives treasure_hunt $S8
run steer_L31_a8_n250 --n 250 --scenarios main $S8
run steer_L31_a8_mir_n250 --n 250 $MIR --scenarios main $S8
echo "=== qwen38 L31 a8 cell complete ==="
