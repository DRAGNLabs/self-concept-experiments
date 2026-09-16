#!/bin/bash --login
#SBATCH --job-name=axis-olmo-smoke
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# End-to-end plumbing check for OLMo 3 support on one GPU. OLMo-3-7B-Think generates
# (with thinking on, so responses carry <think>...</think> blocks) and provides the
# activations, exercising the OlmoModelSpecifics path (ChatML turn parsing, eos
# terminator, char-level think stripping, get_layers, registry). A cheap non-thinking
# Qwen judge scores adherence (a Think model would spend its token budget reasoning
# instead of emitting a 0-3 score). Writes to its own scratch subtree.

source ../../activate.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1  # compute nodes run offline; weights are prefetched
# vLLM's ZMQ IPC socket defaults to tempfile.gettempdir(), which on dw nodes resolves
# to the (long) CWD and blows past the 107-char Unix-socket limit. Pin it short.
export VLLM_RPC_BASE_PATH=/tmp
nvidia-smi

MODEL="allenai/Olmo-3-7B-Think"
JUDGE="Qwen/Qwen2.5-0.5B-Instruct"
ROLES="[default,accountant,comedian]"
SMOKE="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/olmo-smoke"

python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count 8 --run.roles "$ROLES" --run.thinking true --run.max_tokens 1024 \
    --run.output_dir "$SMOKE/responses"

python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.layers 0,1 --run.roles "$ROLES" \
    --run.responses_dir "$SMOKE/responses" --run.output_dir "$SMOKE/activations"

python pipeline/3_judge.py --config configs/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES" \
    --run.responses_dir "$SMOKE/responses" --run.output_dir "$SMOKE/scores"

python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count 1 \
    --run.activations_dir "$SMOKE/activations" --run.scores_dir "$SMOKE/scores" \
    --run.output_dir "$SMOKE/vectors"

python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$SMOKE/vectors" --run.output results/axis_olmo_smoke.pt

echo "=== OLMo smoke done: results/axis_olmo_smoke.pt ==="
