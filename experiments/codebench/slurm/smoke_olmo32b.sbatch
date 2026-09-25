#!/bin/bash --login
#SBATCH --job-name=ib-olmo32b-smoke
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=100G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# Plumbing check: 2 tasks of impossible_oneoff on OLMo-3.1-32B-Think. Confirms the
# chat template + thinking + code extraction + sandbox scoring all work and that
# the 16384-token budget lets the model finish reasoning and submit code (labels
# should be real: cheat_*/honest_*, not all honest_fail from truncated thinking).
REPO_MAIN="/home/kobyjl/experiments/self-concept-experiments"
source "$REPO_MAIN/activate.sh"
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
nvidia-smi

MODEL="allenai/Olmo-3.1-32B-Think"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/olmo32b_think_smoke"
mkdir -p "$OUT"

python -m selfconcept.codebench.run \
    --model "$MODEL" \
    --scenarios impossible_oneoff \
    --n 2 --max-attempts 3 --max-new-tokens 16384 \
    --out "$OUT" --tag smoke

echo "=== smoke summary ==="
cat "$OUT/smoke_impossible_oneoff_summary.json"
echo "=== smoke done ==="
