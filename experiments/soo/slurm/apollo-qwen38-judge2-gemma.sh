#!/bin/bash --login
#SBATCH --job-name=soo-apollo-judge2-gemma
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
# Audit item 7 (2026-09-18): independent-family check of the Apollo roleplaying
# judge. Every roleplaying number in this study comes from one local
# Qwen2.5-72B stand-in for Apollo's GPT-4o judge, never cross-checked. This
# re-grades the round-8 headline conditions with gemma-4-31B-it -- a different
# family, different post-training -- writing *_graded_g31.jsonl so the primary
# grades are untouched (judge_apollo.py --out-suffix, added with this round).
# gemma-4-31B is itself an evaluated model elsewhere in the study, but these
# are Qwen3.8-27B responses, so it is independent here.
# Score with: python scripts/make_apollo_validation_sample.py --score-judge _graded_g31
D=results/apollo_eval/qwen38_27b
TAGS=${TAGS:-ap_base ap_lora ap_lora_agentic_s0 ap_lora_agentic_s1 ap_lora_agentic_s2 ap_lora_mixed_s0 ap_steer_L31 ap_rand_s0_L31}
files=""
for t in $TAGS; do
    f=$D/${t}_roleplaying_none.jsonl
    [ -f "$f" ] && files="$files $f" || echo "missing: $f"
done
echo "grading:$files"
python scripts/judge_apollo.py --model google/gemma-4-31B-it --responses $files \
    --batch-size 16 --max-new-tokens 256 --out-suffix _graded_g31
echo "=== apollo qwen38 second judge (gemma-4-31B) complete ==="
