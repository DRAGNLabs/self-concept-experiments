#!/bin/bash --login
#SBATCH --job-name=persona-drift-delusion-seed
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:4
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/persona_drift  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

nvidia-smi

python -m selfconcept.transcript_generation.generate_seed \
    --config configs/delusion_seed.yaml
