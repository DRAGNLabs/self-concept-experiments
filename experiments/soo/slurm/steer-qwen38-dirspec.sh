#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-dirspec
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
# Qwen3.8-27B direction-specificity sweep. Hardening (13716542) found the
# pilot's 100% cells are in the direction-agnostic regime: at L23 a8
# random matched-norm vectors give main 90/98 orig and 68/100 mirrored
# (base 0/0), at L31 a16 92-94 / 88-92, all clean single-word true rooms;
# -v at L23 a8 is 26/6, at L31 a16 66/100. Same picture as gemma-4-31B
# above a24 and Qwen2.5-72B's orig orientation. gemma-4-31B's
# direction-specific window (a16: real 98 vs rand 18/4) sat just below
# its agnostic regime, so sweep below the working alpha: real, random
# s0/s1 and -v in both orientations at L23 a3-a6 and L31 a6-a12. Verdict
# rule as for Kimi/Qwen2.5: mirrored real >> mirrored random at some alpha
# = direction-specific window; otherwise the model's steering verdict is
# "agnostic flip", and the LoRA (100/100 x 3 seeds) carries the model.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/steering_eval/qwen38_27b
MIR="--data data/eval_mirrored"
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" --scenarios main "$@"
}
for cell in "23 3" "23 4" "23 5" "23 6" "31 6" "31 8" "31 10" "31 12"; do
    set -- $cell; L=$1; A=$2
    S="--steer-vectors $VEC --steer-layer $L"
    for orient in "" "_mir"; do
        D=""; [ "$orient" = "_mir" ] && D="$MIR"
        run "steer_L${L}_a${A}${orient}" $D $S --steer-alpha $A
        run "steer_L${L}_rand0_a${A}${orient}" $D $S --steer-alpha $A --steer-random-seed 0
        run "steer_L${L}_rand1_a${A}${orient}" $D $S --steer-alpha $A --steer-random-seed 1
        run "steer_L${L}_neg_a${A}${orient}" $D $S --steer-alpha -$A
    done
done
echo "=== qwen38 direction-specificity sweep complete ==="
