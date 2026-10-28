#!/bin/bash
#SBATCH --job-name=schemer-gemma31-fitcheck
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis
# One-off: does the gemma-4-31B-it steered engine fit on an 80 GB A100 at max-model-len 40960? Tries several
# gpu_memory_utilization / multimodal-budget settings with --probe (engine build + one answer, then exit).
set -uo pipefail
export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"enable_thinking":true}' VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
fi
export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh
S=$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b
MM='{"limit_mm_per_prompt": {"image": 0}}'
for cfg in "0.95|$MM" "0.97|$MM" "0.98|$MM" "0.98|{}"; do
    gmu=${cfg%%|*}; ek=${cfg#*|}
    echo "########## try gpu_memory_utilization=$gmu engine_kwargs=$ek ##########"
    python scripts/steered_rh_vllm.py --model google/gemma-4-31B-it --scenarios ib_band11 --data $S/rh/band11/data \
        --max-model-len 40960 --max-new-tokens 256 --top-p 0.95 --top-k 64 --gpu-memory-utilization "$gmu" --engine-kwargs "$ek" \
        --direction $S/full/candidates_with_cot_fic/persona_contrast.pt --layer-norms $S/rh/smoke/norms/layer_norms.pt \
        --steer-mode add --steer-layer 30 --steer-alpha -0.02 --out /tmp/fitcheck_$SLURM_JOB_ID --tag fit \
        --probe "Name three prime numbers." 2>&1 | grep -E "Available KV cache memory|Model loading took|ValueError|OutOfMemory|Traceback|GPU KV cache size|Maximum concurrency|probe" | head -8
    echo "########## exit status ${PIPESTATUS[0]} for gmu=$gmu ek=$ek ##########"
done
