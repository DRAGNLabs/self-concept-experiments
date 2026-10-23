#!/bin/bash --login
#SBATCH --job-name=schemer-qwen38
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Expanded schemer-axis extraction for Qwen3.8-27B.  The default role list is
# the fit personas from the completed gptoss120b_extension/full run, plus the
# held-out persona groups used for the separation check and neutral_fic, the
# ordinary fictional-character costume control.  Every role receives all 100
# questions; vectors are fitted only on the 64 train questions.
#
# A Qwen 3.8 27B BF16 checkpoint plus the 12k context is too close to the
# 80-GB A100 limit.  This uses one B200 on cs3, selected with gpu_pick.py and
# approved by the user on 2026-10-23.  Array tasks own disjoint role files and
# may safely share ROOT.  A requeued task resumes existing files.
# Usage (from the repo root):
#   sbatch --array=0-3 --export=ALL,ROOT=<dir> experiments/schemer-axis/slurm/extract_roles_qwen38.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS="${QUESTIONS:-100}"
MIN_COUNT="${MIN_COUNT:-50}"
MAXTOK=8192
MAXLEN=12288
JUDGE_MAXLEN=32768
ACT_BATCH=4
ROLES="${ROLES:-default neutral_fic schemer_fic deceiver_fic cheater_fic con_artist_fic sycophant_fic spin_doctor_fic infiltrator_fic sandbagger_fic straight_shooter_fic rule_follower_fic transparent_negotiator_fic candid_friend_fic scrupulous_scientist_fic whistleblower_fic loyal_steward_fic conscientious_craftsperson_fic diligent_worker_fic embezzler_fic impostor_fic gaslighter_fic honest_bookkeeper_fic humble_expert_fic accountable_owner_fic}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ROLES="$(echo "$ROLES" | tr ' ' '\n' | awk -v n="$SLURM_ARRAY_TASK_COUNT" -v i="$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))" '(NR - 1) % n == i' | tr '\n' ' ')"
fi
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"
ROLES_DIR="../schemer-axis/data/roles"
QUESTIONS_FILE="../schemer-axis/data/questions.jsonl"
ROOT="${ROOT:?set ROOT to the output root}"
echo "model $MODEL roles $ROLES questions $QUESTIONS root $ROOT"

# uv selects the project .venv; activate.sh additionally builds CUDA_HOME from
# its CUDA-13 wheel, which FlashInfer needs when it JIT-compiles B200 kernels.
unset SELFCONCEPT_VENV
source ../../activate.sh
PYTHON=(uv run python)
TRAIN_QUESTIONS="$("${PYTHON[@]}" -c 'import json, sys; print([q["id"] for q in map(json.loads, open(sys.argv[1])) if q["split"] == "train"])' "$QUESTIONS_FILE")"

echo "########## Generate ##########"
"${PYTHON[@]}" scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.85 \
    --run.question_count "$QUESTIONS" --run.temperature 1.0 --run.top_p 1.0 --run.thinking true \
    --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
    --run.roles_dir "$ROLES_DIR" --run.questions_file "$QUESTIONS_FILE" --run.roles "$ROLES_BRACKET" --run.output_dir "$ROOT/responses"

echo "########## Judge ##########"
"${PYTHON[@]}" scripts/pipeline/3_judge.py --config configs/pipeline/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$JUDGE_MAXLEN" --run.roles_dir "$ROLES_DIR" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/scores"

for VARIANT in response_only with_cot; do
    if [ "$VARIANT" = with_cot ]; then INCLUDE_COT=true; else INCLUDE_COT=false; fi
    echo "########## Extract ($VARIANT, include_cot=$INCLUDE_COT) ##########"
    "${PYTHON[@]}" scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml \
        --run.model "$MODEL" --run.include_cot "$INCLUDE_COT" --run.thinking true --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
        --run.roles "$ROLES_BRACKET" \
        --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_$VARIANT"
    "${PYTHON[@]}" scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count "$MIN_COUNT" --run.question_indices "$TRAIN_QUESTIONS" --run.roles "$ROLES_BRACKET" \
        --run.activations_dir "$ROOT/act_$VARIANT" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_$VARIANT"
done

echo "=== schemer-qwen38 done: $ROOT ==="
