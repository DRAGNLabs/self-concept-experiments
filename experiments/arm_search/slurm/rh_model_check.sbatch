#!/bin/bash --login
#SBATCH --job-name=rh-model-check
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=m12,m9
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# Fit health and calibration of the RH ladder model on a search's episodes (selfconcept.arm_search.rh.model_check).
# 4 PyMC chains, so 4 CPUs.
# Usage (from repo root): sbatch experiments/arm_search/slurm/rh_model_check.sbatch --search <search yaml> <report md>
source activate.sh
# Concurrent model-check jobs would race on a shared compile cache in NFS ("Stale file handle"): compile node-locally
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor-${SLURM_JOB_ID}"
python -m selfconcept.arm_search.rh.model_check "$@"
