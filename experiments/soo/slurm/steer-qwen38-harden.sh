#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen38-harden
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
# Qwen3.8-27B steering hardening. Pilot (13707587) main honest %: L23 a4 52
# / a8 100, L24 a8 68, L31 a8 80 / a16 100, L32 a8 84 / a16 90, everything
# else <=22; all working cells answer with clean single-word rooms. Two
# candidate cells, L23 a8 (36% depth, full attn) and L31 a16 (48%, full
# attn), get the gemma-4 treatment: mirrored orientation (base_mir 0/0/100
# already from 13710713), other scenarios, random matched-norm vectors
# (two seeds, both orientations -- mirrored is the certification that
# counts on Qwen-lineage models, orig was contaminated on Qwen2.5-72B),
# -v, dose points past the working alpha for the damage edge, n=250 on
# the headline cell.
MODEL=Qwen/Qwen3.8-27B
VEC=results/steering/qwen38_27b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/qwen38_27b
MIR="--data data/eval_mirrored"
run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}
for cell in "23 8" "31 16"; do
    set -- $cell; L=$1; A=$2
    S="--steer-vectors $VEC --steer-layer $L"
    run "steer_L${L}_a${A}_mir" $MIR --scenarios $SCENARIOS $S --steer-alpha $A
    run "steer_L${L}_a${A}" --scenarios perspectives treasure_hunt $S --steer-alpha $A
    for seed in 0 1; do
        run "steer_L${L}_rand${seed}_a${A}" --scenarios main $S --steer-alpha $A --steer-random-seed $seed
        run "steer_L${L}_rand${seed}_a${A}_mir" $MIR --scenarios main $S --steer-alpha $A --steer-random-seed $seed
    done
    run "steer_L${L}_neg_a${A}" --scenarios main $S --steer-alpha -$A
    run "steer_L${L}_neg_a${A}_mir" $MIR --scenarios main $S --steer-alpha -$A
done
# Damage edge past the working alpha
for A in 12 16 24; do
    run "steer_L23_a${A}" --scenarios main --steer-vectors $VEC --steer-layer 23 --steer-alpha $A
done
for A in 24 32; do
    run "steer_L31_a${A}" --scenarios main --steer-vectors $VEC --steer-layer 31 --steer-alpha $A
done
# n=250 on the headline cell, both orientations
python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only --n 250 --out "$OUT" \
    --tag steer_L23_a8_n250 --scenarios main --steer-vectors $VEC --steer-layer 23 --steer-alpha 8
python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only --n 250 --out "$OUT" $MIR \
    --tag steer_L23_a8_mir_n250 --scenarios main --steer-vectors $VEC --steer-layer 23 --steer-alpha 8
echo "=== qwen38 steering hardening complete ==="
