#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-l31
#SBATCH --time=06:00:00
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
# Qwen3.8-27B headline steering cell L31 a10 (direction-specificity sweep
# 13719042: real 100/100 vs random 18/44 orig, 10/6 mirrored; -v 98/100 --
# axis-specific, sign-agnostic, the gemma-4-12B pattern; L23's window is
# sign-specific but random-leaky). Completes the cell: TH/perspectives in
# both orientations, a third random seed both orientations, n=250 both
# orientations, and capabilities at a10 (13716553 ran a8@L23 and a16@L31).
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/steering_eval/qwen38_27b
MIR="--data data/eval_mirrored"
S="--steer-vectors $VEC --steer-layer 31 --steer-alpha 10"
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --out "$OUT" --tag "$tag" "$@"
}
run steer_L31_a10 --n 50 --scenarios perspectives treasure_hunt $S
run steer_L31_a10_mir --n 50 $MIR --scenarios perspectives treasure_hunt $S
run steer_L31_rand2_a10 --n 50 --scenarios main $S --steer-random-seed 2
run steer_L31_rand2_a10_mir --n 50 $MIR --scenarios main $S --steer-random-seed 2
run steer_L31_a10_n250 --n 250 --scenarios main $S
run steer_L31_a10_mir_n250 --n 250 $MIR --scenarios main $S
python scripts/caps_steered.py $MODEL $VEC 31 10 results/capabilities/steer_qwen38_27b_L31_a10.json
echo "=== qwen38 L31 a10 cell complete ==="
