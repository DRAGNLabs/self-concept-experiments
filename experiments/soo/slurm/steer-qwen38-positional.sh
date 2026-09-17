#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-pos
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
# Qwen3.8-27B positional ("conditional") steering sweep, main scenario,
# both orientations, n=50. The certified full-position cell L31 a10 flips
# the SOO task (100/100) but any matched-norm offset at a>=8 collapses
# multi-turn coding (round 6b), i.e. the dose that flips the task is paid on
# every prompt token too. --steer-positions response adds the vector only
# from the current assistant turn's header (the template's generation
# prompt, where the 'last' vectors were read) through the generated tokens;
# prompt is the complementary diagnostic. Real +v a8-a32 (the response-only
# dose may need to exceed the full-position a10), random s0 and -v at three
# alphas for the direction/sign controls, prompt-only at a10.
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
S="--steer-vectors $VEC --steer-layer 31"
for orient in "" "_mir"; do
    D=""; [ "$orient" = "_mir" ] && D="$MIR"
    for A in 8 10 12 16 24 32; do
        run "steer_L31_resp_a${A}${orient}" $D $S --steer-alpha $A --steer-positions response
    done
    for A in 10 16 32; do
        run "steer_L31_resp_rand0_a${A}${orient}" $D $S --steer-alpha $A --steer-random-seed 0 --steer-positions response
    done
    for A in 10 16; do
        run "steer_L31_resp_neg_a${A}${orient}" $D $S --steer-alpha -$A --steer-positions response
    done
    run "steer_L31_prompt_a10${orient}" $D $S --steer-alpha 10 --steer-positions prompt
done
echo "=== qwen38 positional sweep complete ==="
