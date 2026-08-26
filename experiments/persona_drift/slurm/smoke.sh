#!/bin/bash --login
#SBATCH --job-name=persona-drift-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/persona_drift  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

export HF_HUB_OFFLINE=1

nvidia-smi

python -m selfconcept.transcript_generation.generate_delusion_seed \
    --target-model Qwen/Qwen2.5-0.5B-Instruct --auditor-model Qwen/Qwen2.5-0.5B-Instruct \
    --turns 4 --gpu-memory-utilization 0.45 --max-model-len 4096 \
    --output results/smoke.json
