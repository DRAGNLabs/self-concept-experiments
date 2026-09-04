#!/bin/bash --login
#SBATCH --job-name=soo-steer-kimi-negwin
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:3
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

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
# Sign discriminator inside the Kimi window. The mirrored-window round
# (13584419) proved the a16 window is direction-real in both
# orientations (real 67/80 vs rand 21/20 % honest, orig/mir), so the
# cell is specific — but direction- vs axis-specific needs -v INSIDE
# the window, and -v was only ever run at a24/a32 where the generic
# fragility channel (rand 62-100) and think-truncation bias make it
# unreadable (-v a24 mir: 32/50 think blocks retained, 16/50
# truncated). If -v(a16) sits at rand levels (~20%), Kimi is
# direction-specific; if it flips like +v (~70-80%), axis-specific
# like the 12B. Also thicken the orig a16 random distribution with a
# second seed. All 512 tokens; offline post-think reclass before use.
MODEL=moonshotai/Kimi-Dev-72B
VEC=results/steering/kimi_dev_72b.pt
OUT=results/steering_eval/kimi_dev_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --max-new-tokens 512 --scenarios main \
        --n 50 --tag "$tag" "$@"
}

run neg_L24_a16_long --steer-vectors $VEC --steer-layer 24 --steer-alpha -16
run neg_L24_a16_long_mir --data data/eval_mirrored \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha -16
run rand_s1_L24_a16_long --steer-vectors $VEC --steer-layer 24 --steer-alpha 16 --steer-random-seed 1

echo "=== kimi-dev-72b neg window complete ==="
