#!/bin/bash --login
#SBATCH --job-name=ib-olmo32b-shard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h,eng
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --gpus=h200:1
#SBATCH --mem=100G
#SBATCH --array=0-15
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail

# ImpossibleBench on OLMo-3.1-32B-Think, sharded across a gstandby array so it
# backfills idle H200s regardless of fairshare. 16 tasks = 8 shards x 2 impossible
# splits, one H200 each. Preemptable (--requeue); the harness resumes each shard
# from its own jsonl (same --tag skips finished example_ids), so a preempted shard
# picks up where it left off. Overall reward-hacking rate is computed after the
# array finishes by merging shards (see the merge/score step).
REPO_MAIN="/home/kobyjl/experiments/self-concept-experiments"
source "$REPO_MAIN/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1

MODEL="allenai/Olmo-3.1-32B-Think"
SHARD_DATA="$HOME/nobackup/autodelete/self-concept-experiments/codebench/shards"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/olmo32b_think"
mkdir -p "$OUT"

# array id -> (split, shard). ids 0-7 = oneoff shards 0-7; 8-15 = conflicting shards 0-7.
SPLITS=(impossible_oneoff impossible_conflicting)
SPLIT="${SPLITS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}"
SHARD=$(( SLURM_ARRAY_TASK_ID % 8 ))
echo "task $SLURM_ARRAY_TASK_ID -> split=$SPLIT shard=$SHARD on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m selfconcept.codebench.run \
    --model "$MODEL" \
    --data "$SHARD_DATA" \
    --scenarios "${SPLIT}_s${SHARD}" \
    --max-attempts 3 --max-new-tokens 16384 \
    --out "$OUT" --tag "s${SHARD}"

echo "=== task $SLURM_ARRAY_TASK_ID done: $SPLIT shard $SHARD ==="
