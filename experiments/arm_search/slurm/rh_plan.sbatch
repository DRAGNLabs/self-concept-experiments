#!/bin/bash --login
#SBATCH --job-name=rh-plan
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=m12,m9
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# One round's fit and plan for the reward-hack arm search (selfconcept.arm_search.rh.cli plan); `rh.cli manage`
# submits it with --wait each round. 4 PyMC chains, so 4 CPUs.
# Usage (from repo root): sbatch --wait experiments/arm_search/slurm/rh_plan.sbatch <config>
CONFIG="$1"

source activate.sh
# Rounds plan strictly one after another, so they share one compile cache (no concurrent writers on NFS)
export PYTENSOR_FLAGS="base_compiledir=$HOME/nobackup/autodelete/self-concept-experiments/arm_search_rh/pytensor_cache"
python -m selfconcept.arm_search.rh.cli plan --search "$CONFIG"
