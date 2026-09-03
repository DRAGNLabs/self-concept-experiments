#!/bin/bash --login
#SBATCH --job-name=soo-steer-kimi-window
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
# Last Kimi round. Controls (13573967, post-think reclass) showed L24
# a32 is fully agnostic: rand s0/s1 orig 50/50, rand mir 50/50, -v
# 42/44 — mirroring does not separate, unlike both other 70Bs. The one
# untested escape is a lower-alpha window (Qwen's window sat at a12-16
# where randoms lagged): real dose here is a8 37% / a16 67% / a32 96%,
# randoms unmeasured below a32. Map rand s0 at a8/16/24 + real a24; if
# randoms track real all the way down, the cell verdict is agnostic.
# Also rerun the mirrored n250 anchor the controls job lost to its
# time limit. All 512 tokens; offline post-think reclass before use.
MODEL=moonshotai/Kimi-Dev-72B
VEC=results/steering/kimi_dev_72b.pt
OUT=results/steering_eval/kimi_dev_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --max-new-tokens 512 --scenarios main --tag "$tag" "$@"
}

for A in 8 16 24; do
    run "rand_s0_L24_a${A}_long" --n 50 \
        --steer-vectors $VEC --steer-layer 24 --steer-alpha $A --steer-random-seed 0
done
run steer_L24_a24_long --n 50 \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 24

run steer_L24_a32_long_mir_n250 --n 250 --data data/eval_mirrored \
    --steer-vectors $VEC --steer-layer 24 --steer-alpha 32

echo "=== kimi-dev-72b window probe complete ==="
