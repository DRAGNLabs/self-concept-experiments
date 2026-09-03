#!/bin/bash --login
#SBATCH --job-name=persona-drift-sweep
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/persona_drift  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

python scripts/probe_sweep.py --config "${1:-configs/self_harm.yaml}" --samples "${2:-8}"
