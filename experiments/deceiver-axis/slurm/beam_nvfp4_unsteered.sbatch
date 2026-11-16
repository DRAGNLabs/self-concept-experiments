#!/bin/bash --login
#SBATCH --job-name=beam-nvfp4-unsteered
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=dw87
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:2
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-4
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Unsteered collection for reflection/Beam-1.0-501B-A23B-NVFP4, seeds 0–4.
# Same arm as the Mistral unsteered collection: 45 hard problems, critical_no_carve,
# neutral feedback, 5 stated attempts, temperature 1, top-p 0.95, top-k 64,
# one episode at a time. Two B200s, the layout the fit smoke loaded
# (job 15423823). The shared .venv stays on vLLM 0.28; this job uses .venv-beam.
#
# Two different limits:
#   reasoning_effort minimal is the trained analysis budget, 8 tokens. The
#   template writes "Reasoning effort: 8".
#   max-new-tokens 8192 is the hard stop on the whole reply (analysis plus the
#   answer). That is the limit that keeps a solution from being cut off.
#   Context is 20480, so a long prompt still leaves the 8192 for the reply.
#   The fit smoke's KV pool was about 563k tokens at context 8192.
#
# Do not pin CUDA_VISIBLE_DEVICES to one card: this engine needs both GPUs.
# Recheck gpu_pick.py at submission (--hours 4 --vram-gb 320 --cpus 16
# --mem-gb 256 --cuda 13.0 --no-requeue-ok, non-Blackwell GPUs excluded).
# A task that stops at the limit is resubmitted at the same array index and continues.
#
# Usage, from the repo root:
#   sbatch experiments/deceiver-axis/slurm/beam_nvfp4_unsteered.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_DISABLE_COMPILE_CACHE=1
export SOO_CHAT_KWARGS='{"reasoning_effort":"minimal"}'
export TMPDIR="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$HOME/nobackup/autodelete/tmp/slurm-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"; echo "TMPDIR fallback: $TMPDIR"
fi
nvidia-smi

MODEL="${MODEL:-reflection/Beam-1.0-501B-A23B-NVFP4}"
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"
SEED="${SLURM_ARRAY_TASK_ID:?this script is a 5-task array, seeds 0-4}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/beam-nvfp4/main/unsteered/seed${SEED}}"

export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$(cd ../.. && pwd)/.venv-beam}"
source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "########## Beam NVFP4 unsteered seed $SEED: 45 problems, reasoning minimal (8) ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" \
    --tensor-parallel-size 2 \
    --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 \
    --data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral \
    --max-attempts 5 --state-attempt-budget --max-new-tokens 8192 --max-model-len 20480 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sample-seed "$SEED" \
    --gpu-memory-utilization 0.95 --concurrency 1 \
    --engine-kwargs '{"kv_cache_dtype": "fp8", "max_num_seqs": 4, "max_num_batched_tokens": 8192}' \
    --out "$ROOT" --tag main
echo "seed $SEED took ${SECONDS}s"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob("main_*_reasoning.jsonl")):
    turns = [json.loads(line) for line in path.open()]
    empty = sum(not turn["reasoning"] for turn in turns)
    print(f"{path.name}: {len(turns)} turns, {empty} with empty reasoning, {sum(turn['truncated'] for turn in turns)} truncated")
summary = json.loads((root / "main_impossible_original_summary.json").read_text())
print(f"n={summary['n']} solved={summary['counts']['solved']} failed={summary['counts']['failed']} "
      f"shown_pass={summary['pass_rate']:.3f} hidden_pass={summary['original_pass_rate']:.3f} "
      f"truncated_attempt_rate={summary['truncated_attempt_rate']:.3f}")
PY
echo "=== beam nvfp4 unsteered done: $ROOT ==="
