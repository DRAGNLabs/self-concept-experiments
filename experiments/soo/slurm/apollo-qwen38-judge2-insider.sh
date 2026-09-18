#!/bin/bash --login
#SBATCH --job-name=soo-apollo-judge2-insider
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# gemma-4-31B bf16 (~62 GB) fits one A100 but not with batch-16 long judge
# prompts; two GPUs via device_map=auto. Keep Slurm's set if free, else take
# the two emptiest.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ok=1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ -z "$u" ] || [ "$u" -gt 1000 ]; then ok=0; fi
done
if [ "$ok" = 0 ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -2 | cut -d, -f1 | paste -sd,)
    echo "Assigned set busy or unset; re-pinned to two emptiest"
fi
export CUDA_VISIBLE_DEVICES
echo "Using GPUs $CUDA_VISIBLE_DEVICES"

set -e
# Audit item 7 (2026-09-18), insider half. The round-8 headline regression
# (agentic/mixed adapters conceal 64-89% of insider reports vs base 54) rests
# entirely on judge_insider.py with one local Qwen2.5-72B judge. This re-grades
# the same reports with gemma-4-31B-it into *_graded_g31.jsonl, leaving the
# primary grades untouched (judge_insider.py --out-suffix).
# Score with: python scripts/make_apollo_validation_sample.py --score-judge-insider _graded_g31
D=results/apollo_eval/qwen38_27b
TAGS=${TAGS:-ap_base ap_lora ap_lora_agentic_s0 ap_lora_agentic_s1 ap_lora_agentic_s2 ap_lora_mixed_s0 ap_steer_L31}
files=""
for t in $TAGS; do
    f=$D/${t}_insider_trading_none.jsonl
    [ -f "$f" ] && files="$files $f" || echo "missing: $f"
done
echo "grading:$files"
python scripts/judge_insider.py --model google/gemma-4-31B-it --responses $files \
    --batch-size 16 --max-new-tokens 256 --out-suffix _graded_g31
echo "=== apollo qwen38 second judge insider (gemma-4-31B) complete ==="
