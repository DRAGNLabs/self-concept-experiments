#!/bin/bash --login
#SBATCH --job-name=eg-gptoss120b-failed-shard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h,eng
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --gpus=h200:1
#SBATCH --mem=100G
#SBATCH --array=0-2
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail

# The EvilGenie problems gpt-oss-120b labelled `failed` in the 3-attempt run
# (evilgenie_gptoss120b_shards.sbatch), rerun with 6 attempts and the attempt budget stated in the
# task and feedback messages. 3 contiguous shards, one task each; same resume and TIMEOUT behavior as
# that script. Reasoning goes to the per-shard sidecar s<k>_evilgenie_failed_s<k>_reasoning.jsonl.
# Usage (from repo root): sbatch experiments/codebench/slurm/evilgenie_gptoss120b_failed_shards.sbatch [n]
# With n, runs only the first n problems of each shard into a separate _smoke directory, e.g.
#   sbatch --array=0 --time=01:00:00 <this file> 1
# TEMPERATURE above 0 samples each turn with a seed fixed by SAMPLE_SEED, problem and turn, into a
# _t<TEMPERATURE>_seed<SAMPLE_SEED> directory, e.g.
#   sbatch --export=ALL,TEMPERATURE=1.0,SAMPLE_SEED=1 <this file>
N="${1:-}"
TEMPERATURE="${TEMPERATURE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
N_SHARDS=3
REPO_MAIN="$SLURM_SUBMIT_DIR"
source "$REPO_MAIN/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1

MODEL="openai/gpt-oss-120b"
CODEBENCH="$HOME/nobackup/autodelete/self-concept-experiments/codebench"
SHARD_DATA="$CODEBENCH/shards"
PREVIOUS_RUN="$CODEBENCH/gptoss120b_evilgenie"
if [ "$TEMPERATURE" = 0 ]; then SAMPLING=""; else SAMPLING="_t${TEMPERATURE}_seed${SAMPLE_SEED}"; fi
OUT="$CODEBENCH/gptoss120b_evilgenie_failed_6stated${SAMPLING}${N:+_smoke}"
SHARD=$SLURM_ARRAY_TASK_ID
SHARD_FILE="$SHARD_DATA/evilgenie_failed_s${SHARD}.jsonl"
mkdir -p "$SHARD_DATA" "$OUT"

if [ ! -f "$SHARD_FILE" ]; then
    python - "$REPO_MAIN/benchmarks/codebench/data/evilgenie.jsonl" "$PREVIOUS_RUN" "$SHARD_FILE" "$SHARD" "$N_SHARDS" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

source_path, previous_run, shard_path, shard, n_shards = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
failed_example_ids = {
    record["example_id"]
    for records_path in previous_run.glob("s*_evilgenie.jsonl")
    for record in map(json.loads, records_path.open())
    if record["label"] == "failed"
}
lines = [line for line in open(source_path) if json.loads(line)["example_id"] in failed_example_ids]
assert len(lines) == len(failed_example_ids), (len(lines), len(failed_example_ids))
shard_size = math.ceil(len(lines) / n_shards)
with open(shard_path + ".tmp", "w") as shard_file:
    shard_file.writelines(lines[shard * shard_size : (shard + 1) * shard_size])
os.replace(shard_path + ".tmp", shard_path)
PY
fi

echo "task $SLURM_ARRAY_TASK_ID -> shard $SHARD ($(wc -l < "$SHARD_FILE") problems) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

python -m selfconcept.codebench.run \
    --model "$MODEL" --backend vllm-harmony --tensor-parallel-size "$N_GPUS" \
    --data "$SHARD_DATA" \
    --scenarios "evilgenie_failed_s${SHARD}" \
    --max-attempts 6 --state-attempt-budget --max-new-tokens 16384 \
    --temperature "$TEMPERATURE" --sample-seed "$SAMPLE_SEED" \
    --out "$OUT" --tag "s${SHARD}" ${N:+--n "$N"}

echo "=== task $SLURM_ARRAY_TASK_ID done: shard $SHARD ==="
