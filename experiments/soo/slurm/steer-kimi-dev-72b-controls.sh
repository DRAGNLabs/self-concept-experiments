#!/bin/bash --login
#SBATCH --job-name=soo-steer-kimi-controls
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# Three-GPU variant of the usual re-pin: 72B bf16 (~145 GB) shards across
# three GPUs via device_map=auto (a pair leaves <15 GB headroom). Keep
# Slurm's set if all look free; otherwise take the three emptiest.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -3 | cut -d, -f1 | paste -sd,)
    echo "Assigned set busy or unset; re-pinned to three emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Controls for the longgen rerun (13573571), which established a genuine
# deceptive baseline post-think (main 28/44 answered deceptive, TH 48/48)
# and a big steering flip at L24 a32 (49/50 orig, 50/50 mir). All at 512
# tokens; all outputs get the offline post-think reclassification before
# any number is used (inline labels count think-text room mentions).
# Direction specificity: matched-norm randoms both orientations + -v.
# Also TH under steering, a coarse dose (a8/a16), and n250 anchors for
# the steered cell (cheap: steering suppresses think, responses terse).
MODEL=moonshotai/Kimi-Dev-72B
VEC=results/steering/kimi_dev_72b.pt
OUT=results/steering_eval/kimi_dev_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --max-new-tokens 512 --tag "$tag" "$@"
}

for S in 0 1; do
    run "rand_s${S}_L24_a32_long" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 24 --steer-alpha 32 --steer-random-seed $S
done
run rand_s0_L24_a32_long_mir --n 50 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32 --steer-random-seed 0

run neg_L24_a32_long --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha -32

run steer_L24_a32_long_th --n 50 --scenarios treasure_hunt \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32
run steer_L24_a32_long_th_mir --n 50 --data data/eval_mirrored --scenarios treasure_hunt \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32

for A in 8 16; do
    run "steer_L24_a${A}_long" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 24 --steer-alpha $A
done

run steer_L24_a32_long_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32
run steer_L24_a32_long_mir_n250 --n 250 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32

echo "=== kimi-dev-72b controls complete ==="
