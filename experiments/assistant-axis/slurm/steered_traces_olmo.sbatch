#!/bin/bash --login
#SBATCH --job-name=axis-steered-olmo
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Step 6 role-susceptibility steering traces for Olmo-3.1-32B-Think (Section 3.2.1).
# Usage (from repo root):
#   full:  sbatch slurm/steered_traces_olmo.sbatch <all_tokens|response_only>
#   pilot: sbatch slurm/steered_traces_olmo.sbatch <all_tokens|response_only> data/steer_prompts_pilot.jsonl _pilot
# Reuses the precomputed 64-layer axis + per-layer norms in scratch; steers at the middle
# layer (32) with a coarse fraction-of-norm coefficient sweep. Resumable: reruns skip roles
# whose output .jsonl already exists. HF forward-hook job -> single GPU, no vLLM.

VARIANT="${1:?pass axis variant: all_tokens or response_only}"
PROMPTS="${2:-data/steer_prompts.jsonl}"
TAG="${3:-}"  # e.g. _pilot -> separate output dir
SCRATCH="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis"
AXIS="$SCRATCH/olmo32b-thinking/axis_${VARIANT}.pt"
OUT="$SCRATCH/steered${TAG}_${VARIANT}"

source ../../activate.sh  # uv venv + cluster fixes (repo default); --chdir puts us in experiments/assistant-axis
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1  # compute nodes run offline; weights are prefetched
nvidia-smi

python pipeline/6_steered_traces.py --config configs/6_steered_traces.yaml \
  --run.axis_path "$AXIS" \
  --run.system_prompts_file "$PROMPTS" \
  --run.output_dir "$OUT"

echo "=== Done: $VARIANT steered traces ($PROMPTS) in $OUT ==="
