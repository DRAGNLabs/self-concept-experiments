#!/bin/bash --login
#SBATCH --job-name=schemer-szcap-probe
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs,cs2
#SBATCH --gpus-per-node=1
#SBATCH --qos=cs
#SBATCH --no-requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail
# Size-matched random-direction cap probe (PLAN.md "gpt-oss random-direction p50 cap probe", size amendment 2026-11-10).
# The size-matched thresholds can only be computed once the random directions' unsteered projection sidecars exist
# (cap_thresholds.sbatch), so this job is submitted afterok that job and computes them itself, then runs main_run.sbatch
# unchanged (same recipe, same array-task -> arm / seed mapping, same done marker). Every task recomputes every pair (JSON
# arithmetic, deterministic, atomic write), so each log carries the size table.
# Usage: sbatch --dependency=afterok:<tau job> --array=... \
#   --export=ALL,REF_SIDECAR=<proj.jsonl>,REF_TAU=<{layer: tau}.json>,MATCH="<target proj.jsonl>=<out.json> ...",ARMS=<tsv>,ROOT=<dir>[,NSEEDS=..,...] \
#   experiments/deceiver-axis/slurm/size_matched_cap_probe.sbatch
REF_SIDECAR="${REF_SIDECAR:?set REF_SIDECAR to the projection sidecar of the reference direction}"
REF_TAU="${REF_TAU:?set REF_TAU to the reference cap thresholds}"
MATCH="${MATCH:?set MATCH to target_sidecar=out_json pairs}"
for pair in $MATCH; do
    OPENSSL_CONF=/dev/null "$SLURM_SUBMIT_DIR/.venv-313/bin/python" scripts/cap_size_match.py \
        --ref-sidecar "$REF_SIDECAR" --ref-tau "$REF_TAU" --target-sidecar "${pair%%=*}" --out "${pair#*=}"
done
exec bash --login slurm/main_run.sbatch
