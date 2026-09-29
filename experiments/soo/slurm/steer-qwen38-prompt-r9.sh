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
# Round 9: the prompt-only vector (L31 a10, --steer-positions prompt) as a
# candidate working steering variant (round 7 part 2: full in-distribution
# effect at 50% original-task pass). Controls it still lacks: prompt-only
# random s0 and -v in-distribution (main, both orientations) and the other
# two in-distribution scenarios for +v; then transfer: Apollo roleplaying and
# insider trading for +v prompt-only and random s0 prompt-only. Judged by
# apollo-qwen38-r9-judge.sh (afterok).
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
OUT=results/steering_eval/qwen38_27b
AP=results/apollo_eval/qwen38_27b
S="--steer-vectors $VEC --steer-layer 31 --steer-positions prompt"
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}
for orient in "" "_mir"; do
    D=""; [ "$orient" = "_mir" ] && D="--data data/eval_mirrored"
    run "steer_L31_prompt_rand0_a10${orient}" $D $S --steer-alpha 10 --steer-random-seed 0 --scenarios main
    run "steer_L31_prompt_neg_a10${orient}"   $D $S --steer-alpha -10 --scenarios main
    run "steer_L31_prompt_a10_all${orient}"   $D $S --steer-alpha 10 --scenarios treasure_hunt perspectives
done
ap() {  # ap <tag> <scenario> <tokens> [args...]
    local tag=$1 scen=$2 tok=$3; shift 3
    echo "=== $tag $scen ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_apollo --scenarios $scen \
        --suffix none --max-new-tokens $tok --out "$AP" --tag "$tag" "$@"
}
ap ap_prompt_L31      roleplaying 256 $S --steer-alpha 10
ap ap_prompt_rand_s0_L31 roleplaying 256 $S --steer-alpha 10 --steer-random-seed 0
ap ap_prompt_L31      insider_trading 600 $S --steer-alpha 10
ap ap_prompt_rand_s0_L31 insider_trading 600 $S --steer-alpha 10 --steer-random-seed 0
echo "=== qwen38 prompt-only round 9 complete ==="
