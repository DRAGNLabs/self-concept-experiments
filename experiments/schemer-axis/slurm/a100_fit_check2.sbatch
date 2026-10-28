#!/bin/bash
#SBATCH --job-name=schemer-gemma31-fitcheck2
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis
# One-off follow-up to a100_fit_check.sbatch: gpu_memory_utilization 0.98 built but five of twelve tasks of job
# 14319437 hit CUDA OOM on a ~6k-token prefill (vLLM 0.28 sizes the KV cache before CUDA-graph capture, so the engine
# was ~1.1 GiB over budget with ~0.5 GiB free at runtime). Try pinning the KV cache to just what one 40960-token
# request needs, chunking prefill at 2048 tokens and capturing only small CUDA graphs, then run a ~9k-token probe.
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
LONG=$(python -c "print('Here is a list of integers: ' + ', '.join(str((i*7919) % 1000) for i in range(3600)) + '. What is the sum of the last three numbers? Answer in one line.')")
echo "long probe: ${#LONG} characters"
# 2048-token chunks are refused for this multimodal model (max_tokens_per_mm_item 2496 > max_num_batched_tokens); use 4096,
# first with the image/video/audio budgets zeroed (skips the multimodal profiling/warmup), then plain.
gmu=0.98
EKA='{"kv_cache_memory_bytes":17900000000,"max_num_batched_tokens":4096,"limit_mm_per_prompt":{"image":0,"video":0,"audio":0},"compilation_config":{"cudagraph_capture_sizes":[1,2,4,8]}}'
EKB='{"kv_cache_memory_bytes":17900000000,"max_num_batched_tokens":4096,"compilation_config":{"cudagraph_capture_sizes":[1,2,4,8]}}'
for EK in "$EKA" "$EKB"; do
    echo "########## try gpu_memory_utilization=$gmu engine_kwargs=$EK ##########"
    python scripts/steered_rh_vllm.py --model google/gemma-4-31B-it --scenarios ib_band11 --data $S/rh/band11/data \
        --max-model-len 40960 --max-new-tokens 256 --top-p 0.95 --top-k 64 --gpu-memory-utilization "$gmu" --engine-kwargs "$EK" \
        --direction $S/full/candidates_with_cot_fic/persona_contrast.pt --layer-norms $S/rh/smoke/norms/layer_norms.pt \
        --steer-mode add --steer-layer 30 --steer-alpha -0.02 --out /tmp/fitcheck_$SLURM_JOB_ID --tag fit \
        --probe "$LONG" "Name three prime numbers." 2>&1 | tee /tmp/fitcheck_$SLURM_JOB_ID.log | grep -E "Available KV cache memory|Model loading took|ValueError|OutOfMemory|Traceback|GPU KV cache size|Maximum concurrency|Actual usage|kv_cache_memory|Chunked prefill|probe" | cut -c1-400 | head -12
    echo "########## exit status ${PIPESTATUS[0]} for ek=$EK ##########"
    grep -E "OutOfMemory|Error" /tmp/fitcheck_$SLURM_JOB_ID.log | head -3; rm -rf /tmp/fitcheck_$SLURM_JOB_ID /tmp/fitcheck_$SLURM_JOB_ID.log
    python - <<'PY'
import json, glob
for f in glob.glob("/tmp/fitcheck_*/probe.jsonl"):
    for line in open(f):
        r = json.loads(line); print("probe:", {k: (str(v)[:120]) for k, v in r.items()})
PY
done
