#!/bin/bash --login
#SBATCH --job-name=axis-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# End-to-end plumbing check on a tiny model + a few roles, one GPU. Writes to a
# separate "smoke" scratch subtree so it never collides with a real run. Note: with
# a 0.5B model the judge may give few score=3 samples, so steps 4-5 can skip contrast
# roles -- this validates that the stages run and hand off, not axis quality.

source ../../activate.sh
nvidia-smi

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
ROLES="[default,accountant,comedian]"
SMOKE="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/smoke"

python scripts/pipeline/1_generate.py --config configs/pipeline/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count 8 --run.roles "$ROLES" --run.output_dir "$SMOKE/responses"

python scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml \
    --run.model "$MODEL" --run.layers 0,1 --run.roles "$ROLES" \
    --run.responses_dir "$SMOKE/responses" --run.output_dir "$SMOKE/activations"

python scripts/pipeline/3_judge.py --config configs/pipeline/3_judge.yaml \
    --run.judge_model "$MODEL" --run.tensor_parallel_size 1 --run.roles "$ROLES" \
    --run.responses_dir "$SMOKE/responses" --run.output_dir "$SMOKE/scores"

python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count 1 \
    --run.activations_dir "$SMOKE/activations" --run.scores_dir "$SMOKE/scores" \
    --run.output_dir "$SMOKE/vectors"

python scripts/pipeline/5_axis.py --config configs/pipeline/5_axis.yaml \
    --run.vectors_dir "$SMOKE/vectors" --run.output results/axis_smoke.pt

echo "=== Smoke done: results/axis_smoke.pt ==="
