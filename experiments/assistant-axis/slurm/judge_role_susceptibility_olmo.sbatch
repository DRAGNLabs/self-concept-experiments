#!/bin/bash --login
#SBATCH --job-name=axis-judge-suscept
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1  # vllm 0.28 can't init on P100 (m9g); pin a modern GPU
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Score role-susceptibility steering traces with the local vLLM judge (Qwen2.5-7B-Instruct,
# TP=1) and render the Figure-4-style category-vs-coefficient plot.
# Usage (from repo root): sbatch slurm/judge_role_susceptibility_olmo.sbatch <all_tokens|response_only> [_tag]
VARIANT="${1:?pass axis variant: all_tokens or response_only}"
TAG="${2:-}"  # e.g. _pilot -> matches steered${TAG}_${VARIANT}
SCRATCH="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis"
TRACES="$SCRATCH/steered${TAG}_${VARIANT}"
JUDGE="$SCRATCH/judgements${TAG}_${VARIANT}.jsonl"
PLOT="results/role_susceptibility${TAG}_${VARIANT}.png"

source ../../activate.sh  # uv venv + cluster fixes (VLLM_USE_FLASHINFER_SAMPLER=0, cu13 LD_LIBRARY_PATH)
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1  # judge weights are prefetched
nvidia-smi

python judge_role_susceptibility.py --run.traces_path "$TRACES" --run.output_path "$JUDGE"
python plot_role_susceptibility.py --run.judgements_path "$JUDGE" --run.output_path "$PLOT"

echo "=== Done: judged $TRACES -> $JUDGE ; plot -> $PLOT ==="
