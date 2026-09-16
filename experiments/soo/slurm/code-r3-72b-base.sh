#!/bin/bash --login
#SBATCH --job-name=soo-code3-72b-base
#SBATCH --time=36:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# Reward-hacking coding round 3 = model screen, 72B base cells. $1 = qwen|kimi,
# $2 = scenario. Same protocol as round 1 (n=40, 3 attempts, greedy) except
# Kimi-Dev-72B gets 4096 new tokens because it thinks in <think> blocks
# before answering on code (0 think blocks in its Apollo base run, but those
# were 512-token answers). bf16 across three A100s via device_map=auto as in
# the Apollo runs, so one job per scenario keeps each under the wall clock.
# +v cells (Qwen L40 a16, Kimi L24 a16 per STEERING_MATRIX) and random
# controls follow only if base hacks. Neither model has an SOO LoRA.
which=${1:?usage: sbatch -J <name> code-r3-72b-base.sh qwen|kimi <scenario>}
scenario=${2:?scenario}
case $which in
    qwen) MODEL=Qwen/Qwen2.5-72B-Instruct; OUT=results/code_eval/qwen25_72b; MAXTOK=2048 ;;
    kimi) MODEL=moonshotai/Kimi-Dev-72B; OUT=results/code_eval/kimi_dev_72b; MAXTOK=4096 ;;
    *) echo "unknown model $which"; exit 1 ;;
esac
N=${N:-40}
python -m selfconcept.soo.evaluate_code --model "$MODEL" --device-map auto \
    --scenarios "$scenario" \
    --n "$N" --max-attempts 3 --max-new-tokens "$MAXTOK" --out "$OUT" --tag base
echo "=== code r3 $which base $scenario complete ==="
