#!/bin/bash --login
#SBATCH --job-name=axis-gptoss120b-thinking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# gpt-oss-120b assistant axis, pooled over final-channel tokens only. gpt-oss always reasons;
# the analysis channel is stored as `thinking` and stays in context during extraction.
# MXFP4 fits one H200 in both vLLM and HF (with the `kernels` package). Every step skips
# roles already on disk, so a standby requeue resumes.
# Smoke run: sbatch --export=ALL,ROLES="default pirate",QUESTIONS=2,MIN_COUNT=1,ROOT=<dir> ...

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS="${QUESTIONS:-40}"
MIN_COUNT="${MIN_COUNT:-5}"
MAXTOK=16384
MAXLEN=20480
JUDGE_MAXLEN=32768
ACT_BATCH=4
ATTENTION="kernels-community/vllm-flash-attn3"  # gpt-oss has no sdpa path; eager materializes every attention score
ROLES="${ROLES:-default accountant comedian chef detective pirate poet scientist lawyer doctor teacher philosopher hacker therapist engineer musician historian}"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"

ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/gptoss120b-thinking}"

source ../../activate.sh

echo "########## Generate ##########"
python scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.temperature 1.0 --run.top_p 1.0 --run.thinking true \
    --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
    --run.roles "$ROLES_BRACKET" --run.output_dir "$ROOT/responses"

echo "########## Judge ##########"
python scripts/pipeline/3_judge.py --config configs/pipeline/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$JUDGE_MAXLEN" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/scores"

echo "########## Extract from final-channel tokens ##########"
python scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot false --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
    --run.attn_implementation "$ATTENTION" --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_response_only"
python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$ROOT/act_response_only" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_response_only"
python scripts/pipeline/5_axis.py --config configs/pipeline/5_axis.yaml \
    --run.vectors_dir "$ROOT/vectors_response_only" --run.output "$ROOT/axis_response_only.pt"

echo "=== gptoss120b_thinking done: $ROOT/axis_response_only.pt ==="
