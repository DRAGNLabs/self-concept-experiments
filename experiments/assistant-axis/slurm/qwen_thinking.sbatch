#!/bin/bash --login
#SBATCH --job-name=axis-qwen-thinking
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Qwen3-32B thinking sweep on the PORTED pipeline (greedy, so the three axes differ only
# by the thinking treatment, not sampling noise). Three configs:
#   a) thinking during generation AND extraction (include_cot=true over the CoT tokens)
#   b) thinking during generation only; extraction over response tokens (include_cot=false)
#   c) no thinking at all
# a and b share ONE thinking generation and differ only at activation extraction, so we
# generate twice (thinking on / off) and extract three times.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"  # shared cache (offline on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="Qwen/Qwen3-32B"
JUDGE="Qwen/Qwen2.5-7B-Instruct"
QUESTIONS=40
MIN_COUNT=5
MAXLEN=4096
ROLES="default accountant comedian chef detective pirate poet scientist lawyer doctor teacher philosopher hacker therapist engineer musician historian"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"

ROOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/verify-port/qwen"
THINK="$ROOT/think"
NOTHINK="$ROOT/nothink"
rm -rf "$ROOT"

source ../../activate.sh

echo "########## Generate WITH thinking ##########"
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.temperature 0 --run.thinking true \
    --run.max_tokens 2048 --run.max_model_len "$MAXLEN" \
    --run.roles "$ROLES_BRACKET" --run.output_dir "$THINK/responses"
python pipeline/3_judge.py --config configs/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$MAXLEN" \
    --run.responses_dir "$THINK/responses" --run.output_dir "$THINK/scores"

echo "########## Config a: extract WITH cot ##########"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot true --run.max_length "$MAXLEN" --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$THINK/responses" --run.output_dir "$THINK/act_cot"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$THINK/act_cot" --run.scores_dir "$THINK/scores" --run.output_dir "$THINK/vectors_cot"
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$THINK/vectors_cot" --run.output "$THINK/axis_cot.pt"

echo "########## Config b: extract WITHOUT cot (response tokens only) ##########"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot false --run.max_length "$MAXLEN" --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$THINK/responses" --run.output_dir "$THINK/act_nocot"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$THINK/act_nocot" --run.scores_dir "$THINK/scores" --run.output_dir "$THINK/vectors_nocot"
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$THINK/vectors_nocot" --run.output "$THINK/axis_nocot.pt"

echo "########## Generate WITHOUT thinking (config c) ##########"
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.temperature 0 --run.thinking false \
    --run.max_tokens 512 --run.max_model_len "$MAXLEN" \
    --run.roles "$ROLES_BRACKET" --run.output_dir "$NOTHINK/responses"
python pipeline/3_judge.py --config configs/3_judge.yaml \
    --run.judge_model "$JUDGE" --run.tensor_parallel_size 1 --run.roles "$ROLES_BRACKET" \
    --run.max_model_len "$MAXLEN" \
    --run.responses_dir "$NOTHINK/responses" --run.output_dir "$NOTHINK/scores"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot false --run.max_length "$MAXLEN" --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$NOTHINK/responses" --run.output_dir "$NOTHINK/activations"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$NOTHINK/activations" --run.scores_dir "$NOTHINK/scores" --run.output_dir "$NOTHINK/vectors"
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$NOTHINK/vectors" --run.output "$NOTHINK/axis.pt"

echo "########## Compare the three axes ##########"
python compare_axes.py \
    --axis "think_gen_cot_ext=$THINK/axis_cot.pt" \
    --axis "think_gen_nocot_ext=$THINK/axis_nocot.pt" \
    --axis "nothink=$NOTHINK/axis.pt" \
    --output results/verify-port/qwen_compare.json

echo "=== qwen_thinking done ==="
