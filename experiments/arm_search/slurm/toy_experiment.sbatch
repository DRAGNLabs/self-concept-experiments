#!/bin/bash --login
#SBATCH --job-name=arm-search-toy
#SBATCH --time=3-00:00:00  # standby TIMEOUT is not requeued, so allow the partition maximum
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=m12,m9
#SBATCH --qos=standby
#SBATCH --requeue  # each policy resumes from its state dir
#SBATCH --array=0-23
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail

# Toy arm search experiment: one array task per (replicate world, policy), task i running replicate i / 3 under
# policy i % 3 (4 PyMC chains per fit, so 4 CPUs). The array range must be 0 to 3 * replicate_count - 1.
# Usage (from repo root):
#   sbatch experiments/arm_search/slurm/toy_experiment.sbatch [config]
# then, once every task is done (cheap enough for the login node):
#   python -m selfconcept.arm_search.toy.experiment_cli summarize --experiment <config>
CONFIG="${1:-experiments/arm_search/toy_experiment_rare.yaml}"

source activate.sh
# PyTensor's compile cache lives under base_compiledir; a shared one in NFS home races across array tasks
# ("Stale file handle"), so each job compiles into node-local scratch.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor-${SLURM_JOB_ID}"
POLICIES=(tiered target_only random)
python -m selfconcept.arm_search.toy.experiment_cli run --experiment "$CONFIG" \
  --policy "${POLICIES[SLURM_ARRAY_TASK_ID % 3]}" $((SLURM_ARRAY_TASK_ID / 3))
