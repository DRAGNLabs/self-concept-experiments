#!/bin/bash --login
#SBATCH --job-name=soo-steer-kimi-mirwin
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
# Mirrored-window adjudication, the true final Kimi round. The window
# probe (13575650) found real-vs-random separation in the orig
# orientation at a16-24 (real 67/98 vs rand s0 21/62 % honest of
# answered) — randoms lag real by one dose step and catch up at a32,
# the same shape Qwen2.5-72B showed before its mirrored round proved
# direction-specificity. Kimi's mirrored rand at a32 was 50/50 (no
# separation at plateau), but plateau agnosticism is true of every
# model; the window itself was never tested mirrored. Run the mirrored
# window: real + rand s0 at a16 and a24, rand s1 + -v at a24. All 512
# tokens; offline post-think reclass before use.
MODEL=moonshotai/Kimi-Dev-72B
VEC=results/steering/kimi_dev_72b.pt
OUT=results/steering_eval/kimi_dev_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --max-new-tokens 512 --scenarios main \
        --n 50 --data data/eval_mirrored --tag "$tag" "$@"
}

for A in 16 24; do
    run "steer_L24_a${A}_long_mir" --steer-vectors $VEC --steer-layer 24 --steer-alpha $A
    run "rand_s0_L24_a${A}_long_mir" --steer-vectors $VEC --steer-layer 24 --steer-alpha $A --steer-random-seed 0
done
run rand_s1_L24_a24_long_mir --steer-vectors $VEC --steer-layer 24 --steer-alpha 24 --steer-random-seed 1
run neg_L24_a24_long_mir --steer-vectors $VEC --steer-layer 24 --steer-alpha -24

echo "=== kimi-dev-72b mirrored window complete ==="
