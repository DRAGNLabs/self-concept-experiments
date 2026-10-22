#!/bin/bash --login
#SBATCH --job-name=goodfire-rh-judge-olmo32b-ib
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

# Goodfire (arXiv 2609.19101) ImpossibleBench judge, minimal-scaffold rubric, over the
# OLMo-3.1-32B-Think ImpossibleBench transcripts, judged by gpt-oss-120b (reasoning high).
# Usage (from repo root): sbatch experiments/goodfire_rh_metrics/slurm/judge_olmo32b_think_impossiblebench.sbatch [n]
# With n, judges only the first n transcripts per split into a separate _smoke_<partition> directory;
# a smoke run needs far less than the full run's 8h: sbatch --time=01:00:00 <this file> 4
# Tensor parallelism spans every allocated GPU, so other GPUs are a command-line override, e.g.
# two A100-80GB (one leaves ~10 GB for KV cache): sbatch -p dw --qos=dw87 --gpus=a100:2 <this file>
N="${1:-}"
TRANSCRIPTS="$HOME/nobackup/autodelete/self-concept-experiments/codebench/olmo32b_think"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/goodfire_rh_metrics/olmo32b_think${N:+_smoke_$SLURM_JOB_PARTITION}"

source activate.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

python -m selfconcept.goodfire_rh_metrics.judge \
    --run.benchmark impossiblebench_minimal_scaffold \
    --run.transcripts_paths "[$TRANSCRIPTS/merged_impossible_oneoff.jsonl, $TRANSCRIPTS/merged_impossible_conflicting.jsonl]" \
    --run.output_dir "$OUT" --run.tensor_parallel_size "$N_GPUS" ${N:+--run.n "$N"}

echo "=== Done: judged -> $OUT ==="
