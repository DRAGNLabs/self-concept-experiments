#!/bin/bash --login
#SBATCH --job-name=goodfire-rh-judge-gptoss120b-eg
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h,eng
#SBATCH --gpus=h200:1  # gpt-oss-120b MXFP4 fits one H200; vllm 0.28 can't init on P100
#SBATCH --qos=standby
#SBATCH --requeue  # the judge resumes from its judged_*.jsonl, one chunk at a time
#SBATCH --mem=100G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# Goodfire (arXiv 2609.19101) judge, EvilGenie file-block rubric, over the gpt-oss-120b EvilGenie
# transcripts (experiments/codebench/slurm/evilgenie_gptoss120b_shards.sbatch), judged by
# gpt-oss-120b (reasoning high). Each shard's records are paired with its reasoning sidecar.
# Usage (from repo root): sbatch experiments/goodfire_rh_metrics/slurm/judge_gptoss120b_evilgenie.sbatch [n]
# With n, judges only the first n transcripts per shard into a separate _smoke_<partition> directory:
#   sbatch --time=01:00:00 <this file> 1
# Tensor parallelism spans every allocated GPU, so 80GB cards are a command-line override, e.g.
#   sbatch -p cs --qos=cs --gpus=a100:2 --exclude=cs-1-2 <this file>
N="${1:-}"
N_SHARDS=8
TRANSCRIPTS="$HOME/nobackup/autodelete/self-concept-experiments/codebench/gptoss120b_evilgenie"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/goodfire_rh_metrics/gptoss120b_evilgenie${N:+_smoke_$SLURM_JOB_PARTITION}"

source activate.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

RECORDS=()
SIDECARS=()
for SHARD in $(seq 0 $(( N_SHARDS - 1 ))); do
    RECORDS+=("$TRANSCRIPTS/s${SHARD}_evilgenie.jsonl")
    SIDECARS+=("$TRANSCRIPTS/s${SHARD}_evilgenie_s${SHARD}_reasoning.jsonl")
done
join_paths () { local IFS=,; echo "[$*]"; }

python -m selfconcept.goodfire_rh_metrics.judge \
    --run.benchmark evilgenie_file_blocks \
    --run.model_specifics harmony \
    --run.transcripts_paths "$(join_paths "${RECORDS[@]}")" \
    --run.reasoning_paths "$(join_paths "${SIDECARS[@]}")" \
    --run.output_dir "$OUT" --run.tensor_parallel_size "$N_GPUS" ${N:+--run.n "$N"}

python -m selfconcept.goodfire_rh_metrics.metrics \
    --run.benchmark evilgenie_file_blocks \
    --run.judged_paths "$(join_paths "$OUT"/judged_s*_evilgenie.jsonl)" \
    --run.output_path "$OUT/metrics_summary.json"

echo "=== Done: judged -> $OUT ==="
