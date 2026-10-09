#!/bin/bash --login
#SBATCH --job-name=schemer-extract
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Schemer-axis role vectors for gpt-oss-120b (experiments/schemer-axis/PLAN.md, Phase 1). Same pipeline and
# settings as slurm/gptoss120b_thinking.sbatch (thinking on, temperature 1, 40 questions x 5 prompts per role,
# Qwen2.5-7B role judge), with the roles read from experiments/schemer-axis/data/roles and the activations
# extracted twice: pooled over final-channel tokens only (act_response_only) and over analysis+final tokens
# (act_with_cot). No axis is computed here; the candidate vectors are built from vectors_*/{role}.pt offline.
# Every step skips roles already on disk, so a standby requeue resumes.
# Smoke run: sbatch --export=ALL,ROLES="default schemer",QUESTIONS=2,MIN_COUNT=1,ROOT=<dir> ...

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS="${QUESTIONS:-40}"
MIN_COUNT="${MIN_COUNT:-50}"
MAXTOK=16384
MAXLEN=20480
JUDGE_MAXLEN=32768
ACT_BATCH=4
ATTENTION="kernels-community/vllm-flash-attn3"  # gpt-oss has no sdpa path; eager materializes every attention score
ROLES="${ROLES:-default schemer deceiver cheater straight_shooter rule_follower}"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"
ROLES_DIR="../schemer-axis/data/roles"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/roles}"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

echo "########## Generate ##########"
python scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.temperature 1.0 --run.top_p 1.0 --run.thinking true \
    --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
    --run.roles_dir "$ROLES_DIR" --run.roles "$ROLES_BRACKET" --run.output_dir "$ROOT/responses"

echo "########## Judge ##########"
python scripts/pipeline/3_judge.py --config configs/pipeline/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$JUDGE_MAXLEN" --run.roles_dir "$ROLES_DIR" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/scores"

for VARIANT in response_only with_cot; do
    if [ "$VARIANT" = with_cot ]; then INCLUDE_COT=true; else INCLUDE_COT=false; fi
    echo "########## Extract ($VARIANT, include_cot=$INCLUDE_COT) ##########"
    python scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml \
        --run.model "$MODEL" --run.include_cot "$INCLUDE_COT" --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
        --run.attn_implementation "$ATTENTION" --run.roles "$ROLES_BRACKET" \
        --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_$VARIANT"
    python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count "$MIN_COUNT" \
        --run.activations_dir "$ROOT/act_$VARIANT" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_$VARIANT"
done

echo "=== schemer-extract done: $ROOT ==="
