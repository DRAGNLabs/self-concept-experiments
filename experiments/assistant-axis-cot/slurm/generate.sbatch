#!/bin/bash --login
#SBATCH --job-name=axis-cot-generate
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../assistant-axis-cot/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Held-out thinking transcripts for the AA x CoT study: the persona roles in
# ../assistant-axis-cot/data/persona_roles.jsonl plus 5 unprompted samples per question, on
# held-out questions. persona_roles.jsonl came from (in experiments/assistant-axis):
#   python scripts/steering/select_steer_roles.py --config configs/steering/select_steer_roles.yaml \
#       --run.top_k 20 --run.least_assistant_like true --run.output ../assistant-axis-cot/data/persona_roles.jsonl
# Sampling settings match the model's axis run (slurm/*_thinking.sbatch).
# Single GPU: [[vllm028-multigpu-node-startup-hang]]. 1_generate.py skips roles already on
# disk, so resubmitting resumes.
#   sbatch --export=ALL,MODEL_KEY=olmo   experiments/assistant-axis-cot/slurm/generate.sbatch
#   sbatch --export=ALL,MODEL_KEY=gptoss experiments/assistant-axis-cot/slurm/generate.sbatch
# Smoke: add QUESTIONS=2,PERSONAS="leviathan tree",RUN=<model>-smoke

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

case "$MODEL_KEY" in
    olmo)
        MODEL="allenai/Olmo-3.1-32B-Think"
        TEMPERATURE=0.6; TOP_P=0.95; MAXTOK=32768; MAXLEN=40960
        RUN="${RUN:-olmo32b-heldout}" ;;
    gptoss)
        MODEL="openai/gpt-oss-120b"
        TEMPERATURE=1.0; TOP_P=1.0; MAXTOK=16384; MAXLEN=20480
        RUN="${RUN:-gptoss120b-heldout}" ;;
    *) echo "MODEL_KEY must be olmo or gptoss" >&2; exit 1 ;;
esac

COT_DIR=../assistant-axis-cot
QUESTIONS="${QUESTIONS:-30}"
PERSONAS="${PERSONAS:-$(jq -r .role_id "$COT_DIR/data/persona_roles.jsonl" | uniq | tr '\n' ' ')}"
PERSONAS_BRACKET="[$(echo $PERSONAS | tr ' ' ',')]"
OUTPUT_DIR="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis-cot/$RUN/responses"

source ../../activate.sh

generate() {
    python scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
        --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
        --run.questions_file "$COT_DIR/data/heldout_questions.jsonl" --run.question_count "$QUESTIONS" \
        --run.temperature "$TEMPERATURE" --run.top_p "$TOP_P" --run.thinking true \
        --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
        --run.output_dir "$OUTPUT_DIR" "$@"
}

echo "########## Unprompted ##########"
generate --run.roles_dir "$COT_DIR/data/roles" --run.roles "[unprompted]"

echo "########## Personas ##########"
generate --run.roles "$PERSONAS_BRACKET"

echo "=== generate done: $OUTPUT_DIR ==="
