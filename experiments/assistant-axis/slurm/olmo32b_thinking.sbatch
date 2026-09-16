#!/bin/bash --login
#SBATCH --job-name=axis-olmo32b-thinking
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# OLMo-3.1-32B-Think sweep: ONE thinking generation, extracted TWO ways.
#   a) all assistant tokens   (include_cot=true  -- thinking + response both count)
#   b) response tokens only  (include_cot=false -- thinking tokens filtered out)
# Single GPU only: [[vllm028-multigpu-node-startup-hang]] means the ported env's vLLM
# 0.28 hangs at engine startup on any >1-GPU node allocation, so TP>1 isn't an option
# here regardless of whether it'd help. A 32B model in bf16 (~60GB) fits comfortably on
# one H200 (141GB). Confirmed on the real 17-role x 40-question sweep (~3400 prompts,
# max response length 8743 tokens, all closed their <think> block well under the
# max_tokens=32768 cap): stage 1 (generation) took ~45min. Stage 2 (activations) needs a
# small --run.batch_size: extracting all 64 layers' hidden states for a long-context 32B
# model materializes (num_layers, batch, seq_len, hidden) in fp32-ish working memory, and
# at the default batch_size=16 (tuned for gemma-2-27b-it's 2048-token max_length) that
# OOMs a single H200 once sequences run into the thousands of tokens.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="allenai/Olmo-3.1-32B-Think"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS=40
MIN_COUNT=5
MAXTOK=32768
MAXLEN=40960
JUDGE_MAXLEN=32768  # Qwen2.5-7B-Instruct's native max_position_embeddings; can't be raised like OLMo's
ACT_BATCH=2  # small batch to avoid OOM extracting 64 layers x long context (see comment above)
ROLES="default accountant comedian chef detective pirate poet scientist lawyer doctor teacher philosopher hacker therapist engineer musician historian"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"

ROOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/olmo32b-thinking"

source ../../activate.sh

echo "########## Generate WITH thinking ##########"
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.temperature 0.6 --run.top_p 0.95 --run.thinking true \
    --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
    --run.roles "$ROLES_BRACKET" --run.output_dir "$ROOT/responses"

echo "########## Judge (shared by both extraction configs) ##########"
python pipeline/3_judge.py --config configs/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$JUDGE_MAXLEN" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/scores"

echo "########## Config a: extract from ALL assistant tokens (include_cot=true) ##########"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot true --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
    --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_all_tokens"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$ROOT/act_all_tokens" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_all_tokens"
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$ROOT/vectors_all_tokens" --run.output "$ROOT/axis_all_tokens.pt"

echo "########## Config b: extract from RESPONSE tokens only (include_cot=false) ##########"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot false --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
    --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_response_only"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$ROOT/act_response_only" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_response_only"
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$ROOT/vectors_response_only" --run.output "$ROOT/axis_response_only.pt"

echo "########## Compare the two axes ##########"
python compare_axes.py \
    --axis "all_tokens=$ROOT/axis_all_tokens.pt" \
    --axis "response_only=$ROOT/axis_response_only.pt" \
    --output results/olmo32b_thinking_compare.json

echo "=== olmo32b_thinking done ==="
