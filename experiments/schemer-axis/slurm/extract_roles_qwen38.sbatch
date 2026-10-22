#!/bin/bash --login
#SBATCH --job-name=schemer-qwen38
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=eng
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Schemer-axis role vectors for Qwen3.8-27B. Same steps as extract_roles_gemma4.sbatch:
# generate, judge, pool answer tokens, pool thinking tokens, write per-role means.
# Thinking stays on. A resubmission skips roles already on disk.
# 52 GB of weights plus a 12k context fits an H200; an 80 GB A100 is tight.
# Usage (from the repo root):
#   sbatch --export=ALL,ROOT=<dir> experiments/schemer-axis/slurm/extract_roles_qwen38.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS="${QUESTIONS:-40}"
# 4_vectors.py keeps only judge score 3. 50 is unreachable with 40 questions, so the
# default here is 1. The Phase 1 gate (50 of 200) needs a rerun with QUESTIONS=200.
MIN_COUNT="${MIN_COUNT:-1}"
MAXTOK=8192
MAXLEN=12288
JUDGE_MAXLEN=32768
ACT_BATCH=4
ROLES="${ROLES:-default schemer_fic deceiver_fic cheater_fic straight_shooter_fic rule_follower_fic neutral_fic}"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"
ROLES_DIR="../schemer-axis/data/roles"
ROOT="${ROOT:?set ROOT to the output root}"
echo "model $MODEL roles $ROLES questions $QUESTIONS root $ROOT"

source ../../activate.sh

echo "########## Generate ##########"
python scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.85 \
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
        --run.model "$MODEL" --run.include_cot "$INCLUDE_COT" --run.thinking true --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
        --run.roles "$ROLES_BRACKET" \
        --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_$VARIANT"
    python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count "$MIN_COUNT" \
        --run.activations_dir "$ROOT/act_$VARIANT" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_$VARIANT"
done

echo "=== schemer-qwen38 done: $ROOT ==="
