#!/bin/bash --login
#SBATCH --job-name=ms4-unsteered
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=dw87
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-11
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Unsteered collection for mistralai/Mistral-Small-4-119B-2603, matching Jansen's
# Muse unsteered arm (muse_glimmer/main/arms.tsv, the unsteered line, run through
# main_run.sbatch on 2026-10-21): all 45 hard problems, seeds 0–11, one array
# task per seed. Task index is the seed. A rerun of the same index resumes;
# finished example ids are skipped.
#
# Recipe, same as that arm: critical_no_carve, neutral feedback, 5 stated
# attempts, temperature 1, top-p 0.95, top-k 64, 16384 new tokens, context
# 20480, one episode at a time, memory fraction 0.95. Muse set reasoning
# strength to low. This template has no low setting; high is the setting that
# leaves a [THINK] trace for the judge, and none would leave the sidecar empty.
#
# Mistral-only engine settings: the native checkpoint is the FP8 Mistral repo
# (a Pixtral wrapper), so the vision tower gets no profiling budget, and the
# sequence cap matches the fit smoke. B200 auto-selects FLASHINFER_MLA, which
# downloads a cubin; compute nodes cannot reach NVIDIA, so the backend is
# TRITON_MLA. One B200 on cs3 / qos cs / account dw87,
# from the 2026-11-12 fit-smoke pick: smaller cards cannot hold the checkpoint.
# Recheck gpu_pick.py before submitting and save that output next to the run.
# Do not submit until the fit smoke has loaded the model.
#
# A Muse seed took about 2.5 h on an H200 and 3.7 h on an A100, with a 5 h
# limit. Eight hours leaves room for high-effort traces on this model. A task
# that stops at the limit is resubmitted at the same array index and continues.
#
# Usage, from the repo root (this submits all 12 seeds; do not run it yet):
#   sbatch experiments/deceiver-axis/slurm/mistral_small4_unsteered.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"high"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
export TMPDIR="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$HOME/nobackup/autodelete/tmp/slurm-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"; echo "TMPDIR fallback: $TMPDIR"
fi
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-mistralai/Mistral-Small-4-119B-2603}"
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"
SEED="${SLURM_ARRAY_TASK_ID:?this script is a 12-task array, one seed per task}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/mistral-small4/main/unsteered/seed${SEED}}"

source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "########## Mistral-Small-4 unsteered seed $SEED: 45 problems ##########"
SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" \
    --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 \
    --data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral \
    --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --max-model-len 20480 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sample-seed "$SEED" \
    --gpu-memory-utilization 0.95 --concurrency 1 \
    --engine-kwargs '{"limit_mm_per_prompt": {"image": 0}, "max_num_seqs": 4, "max_num_batched_tokens": 8192, "attention_backend": "TRITON_MLA"}' \
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
      f"shown_pass={summary['pass_rate']:.3f} hidden_pass={summary['original_pass_rate']:.3f}")
PY
echo "=== ms4 unsteered done: $ROOT ==="
