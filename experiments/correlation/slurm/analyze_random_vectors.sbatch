#!/bin/bash --login
#SBATCH --job-name=rh-random-analysis
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail
export OPENSSL_CONF=/dev/null
export PYTHONPATH="$PWD/.runtime:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
"$PYTHON" -u -m selfconcept.correlation.random_vector_analysis "$OUT"/shard_* \
    --axis "$AXIS" --out "$OUT/analysis" --random-count 256 --seed 1729 --bootstrap 1000
