#!/bin/bash --login
#SBATCH --job-name=axis-cot-project
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis-cot  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Per-token AA projections for one model's transcripts: DATA=axis is the axis-training run's
# responses, DATA=heldout the held-out ones from generate.sbatch.
#   sbatch --export=ALL,MODEL_KEY=olmo,DATA=heldout experiments/assistant-axis-cot/slurm/project.sbatch
# Smoke: add CONVERSATIONS_PER_ROLE=2,RUN=<model>-smoke-project

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"  # harmony vocab; node-local /tmp default is empty offline
nvidia-smi

AXIS_ROOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis"
COT_ROOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis-cot"

case "$MODEL_KEY" in
    olmo)
        MODEL="allenai/Olmo-3.1-32B-Think"; MODEL_DIR=olmo32b; MAXLEN=40960; ATTENTION=null
        AXES="{all_tokens: $AXIS_ROOT/olmo32b-thinking/axis_all_tokens.pt, response_only: $AXIS_ROOT/olmo32b-thinking/axis_response_only.pt}" ;;
    gptoss)
        MODEL="openai/gpt-oss-120b"; MODEL_DIR=gptoss120b; MAXLEN=20480
        ATTENTION="kernels-community/vllm-flash-attn3"  # gpt-oss has no sdpa path; eager materializes every attention score. Hopper only
        AXES="{response_only: $AXIS_ROOT/gptoss120b-thinking/axis_response_only.pt}" ;;
    *) echo "MODEL_KEY must be olmo or gptoss" >&2; exit 1 ;;
esac

case "$DATA" in
    axis) RESPONSES_DIR="$AXIS_ROOT/$MODEL_DIR-thinking/responses" ;;
    heldout) RESPONSES_DIR="$COT_ROOT/$MODEL_DIR-heldout/responses" ;;
    *) echo "DATA must be axis or heldout" >&2; exit 1 ;;
esac

RUN="${RUN:-$MODEL_DIR-$DATA}"
CONVERSATIONS_PER_ROLE="${CONVERSATIONS_PER_ROLE:-null}"

source ../../activate.sh

python scripts/project.py --config configs/project.yaml \
    --run.model "$MODEL" --run.responses_dir "$RESPONSES_DIR" --run.axis_path_by_name "$AXES" \
    --run.max_length "$MAXLEN" --run.attn_implementation "$ATTENTION" \
    --run.conversations_per_role "$CONVERSATIONS_PER_ROLE" \
    --run.output "$COT_ROOT/$RUN/token_projections.pt"
