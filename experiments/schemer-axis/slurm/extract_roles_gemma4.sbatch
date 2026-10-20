#!/bin/bash --login
#SBATCH --job-name=schemer-gemma4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs,cs2
#SBATCH --gpus-per-node=1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Schemer-axis role vectors for a Gemma 4 model (experiments/schemer-axis/PLAN.md, "Gemma 4 pilot"): the same
# pipeline and settings as extract_roles.sbatch (gpt-oss-120b) with MODEL overridable. Thinking is on (the Gemma 4
# specifics inject <|think|> and split the thought channel off the answer); activations are pooled twice, over the
# answer tokens (act_response_only) and over the thought-channel tokens (act_with_cot). Every step skips roles
# already on disk, so a resubmission resumes. Default roles are the six fiction roles + default; add the bare roles
# (schemer deceiver cheater) to see whether Gemma plays a bare persona that gpt-oss refused.
# Usage: sbatch --export=ALL,ROOT=<dir>[,MODEL=google/gemma-4-E4B-it][,ROLES="..."][,QUESTIONS=40][,MIN_COUNT=50][,ACT_BATCH=4][,MAXTOK=8192][,MAXLEN=12288] experiments/schemer-axis/slurm/extract_roles_gemma4.sbatch
# Muse Glimmer (PLAN "Muse Glimmer replication"): MODEL=meta-models/Muse-Glimmer-30B with the same thinking-on settings (its to=self channel
# is always on; the model specifics split it off the answer) and MAXTOK=16384 MAXLEN=20480 for its default reasoning strength.
# Smoke: sbatch --export=ALL,ROLES="default schemer schemer_fic straight_shooter_fic",QUESTIONS=3,MIN_COUNT=1,ROOT=<dir> ...

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi
# cs A100 nodes can expose a sibling task's GPU as well (see main_run.sbatch): pin to the least-used visible GPU.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
# Blackwell (B200): vLLM auto-selects FLASHINFER for the Qwen judge and its JIT build fails against this venv's CUDA
# headers ("CUDA compiler and CUDA toolkit headers are incompatible", job 14272566). TRITON_ATTN works there;
# VLLMGenerator passes the variable on as vLLM's attention_backend argument (Gemma 4 picks TRITON_ATTN anyway).
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q B200; then
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
    echo "Blackwell GPU: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
fi

MODEL="${MODEL:-google/gemma-4-12B-it}"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS="${QUESTIONS:-40}"
MIN_COUNT="${MIN_COUNT:-50}"
MAXTOK="${MAXTOK:-8192}"
MAXLEN="${MAXLEN:-12288}"
JUDGE_MAXLEN=32768
ACT_BATCH="${ACT_BATCH:-4}"
ROLES="${ROLES:-default schemer_fic deceiver_fic cheater_fic straight_shooter_fic rule_follower_fic neutral_fic}"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"
ROLES_DIR="../schemer-axis/data/roles"
ROOT="${ROOT:?set ROOT to the output root}"
echo "model $MODEL roles $ROLES questions $QUESTIONS root $ROOT"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
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

echo "=== schemer-gemma4 done: $ROOT ==="
