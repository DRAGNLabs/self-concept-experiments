#!/bin/bash --login
# BYU ORC policy: repository BYU_ORC_AGENTS.md. $1/$2 are frozen source/protocol.
#SBATCH --job-name=soo-p3fp-prepare
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/home/jansen05/self-concept-experiments/experiments/soo/slurm-logs/%x-%j.out
set -euo pipefail
cd /home/jansen05/self-concept-experiments
export PYTHONPATH="$PWD/experiments/soo/results/study2/part3-addback1-controller-20261029T213251426612Z/src"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/.venv-313/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
.venv-313/bin/python -u "$1" --stage "$PWD" --protocol "$2" --submit
