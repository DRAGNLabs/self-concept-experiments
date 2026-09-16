#!/bin/bash --login
#SBATCH --job-name=soo-steer-qwen25-72b-val
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
# Validation of the pilot (13566237): baseline 0/0 (main/TH), and a32
# lifted L16 88 / L32 98 / L40 100 with clean varied responses (11-14
# distinct rooms), while a8 stayed low (38-60) -- opposite dose profile
# to Llama-2-70b, whose a32 was degenerate. Best cell L40 a32. Controls
# decide direction-specificity: mirrored orientation, TH under steering,
# dose curve, matched-norm randoms (both orientations), -v, n=250
# anchors. Vectors already extracted (results/steering/qwen25_72b.pt).
MODEL=Qwen/Qwen2.5-72B-Instruct
VEC=results/steering/qwen25_72b.pt
OUT=results/steering_eval/qwen25_72b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --device-map auto \
        --out "$OUT" --suffix room_only --tag "$tag" "$@"
}

run baseline_mir --n 50 --data data/eval_mirrored --scenarios main treasure_hunt

run steer_L40_a32_mir --n 50 --data data/eval_mirrored --scenarios main treasure_hunt \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 32
run steer_L40_a32_th --n 50 --scenarios treasure_hunt \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 32

for A in 16 24 48; do
    run "steer_L40_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 40 --steer-alpha $A
done

for S in 0 1; do
    run "rand_s${S}_L40_a32" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 40 --steer-alpha 32 --steer-random-seed $S
done
run rand_s0_L40_a32_mir --n 50 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 32 --steer-random-seed 0

run neg_L40_a32 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha -32

run steer_L40_a32_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 32
run steer_L40_a32_mir_n250 --n 250 --data data/eval_mirrored --scenarios main \
    --steer-vectors $VEC --steer-layer 40 --steer-alpha 32

echo "=== qwen25-72b validation complete ==="
