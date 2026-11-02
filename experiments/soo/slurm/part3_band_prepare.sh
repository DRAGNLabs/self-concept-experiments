#!/bin/bash --login
# BYU ORC policy: repository BYU_ORC_AGENTS.md
#SBATCH --job-name=soo-p3-prepare
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
export PYTHONPATH="$PWD/src"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
/usr/bin/python3.11 -m unittest discover -s tests -p test_soo_part3.py
/usr/bin/python3.11 -u experiments/soo/scripts/part3_band.py --stage "$PWD" \
    --audit "$PWD/experiments/soo/results/study2/part3-existing-audit-20261029" --submit
