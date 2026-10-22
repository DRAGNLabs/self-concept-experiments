#!/bin/bash --login
#SBATCH --job-name=soo-steer-g4-12b-mirspec
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

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
# Mirrored adjudication of the 12B "agnostic-flip" verdict. On both
# Llama-2-70b and Qwen2.5-72B the orig orientation looked (sign-)
# agnostic and the mirrored orientation was decisive: Qwen mirrored at
# L40 a16 gave real 100 vs rand 28/0 and -v 22. The 12B verdict rests
# entirely on orig-orientation controls (-v(a12)=100, rand s0/s1(a12)
# = 68/0); its mirrored controls were never run. Real mirrored is
# already known strong (0 -> 100 at a32). Run the mirrored controls at
# both the band edge (a12) and the plateau (a32).
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
OUT=results/steering_eval/gemma4_12b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --n 50 --out "$OUT" --suffix room_only --scenarios main \
        --data data/eval_mirrored --tag "$tag" "$@"
}

run steer_L19_a12_mir --steer-vectors $VEC --steer-layer 19 --steer-alpha 12

for A in 12 32; do
    for S in 0 1; do
        run "rand_s${S}_L19_a${A}_mir" \
            --steer-vectors $VEC --steer-layer 19 --steer-alpha $A --steer-random-seed $S
    done
    run "neg_L19_a${A}_mir" --steer-vectors $VEC --steer-layer 19 --steer-alpha "-$A"
done

echo "=== g4-12b mirrored specificity complete ==="
