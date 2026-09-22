#!/bin/bash --login
#SBATCH --job-name=ib-olmo32b-think
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --gpus=h200:2
#SBATCH --mem=160G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# ImpossibleBench (Zhong et al. 2025) reward-hacking eval on OLMo-3.1-32B-Think,
# via the repo's self-contained port (selfconcept.codebench). The two *impossible*
# splits are the reward-hacking measurement (passing a suite you can only pass by
# gaming it); overall RH rate = pass_rate averaged over both. Each split runs on
# its own H200 (OLMo 32B bf16 ~= 64 GB, fits one card), greedy, 16384-token budget
# so the Think model can finish <think> then emit code (a small budget silently
# reads as 0% hacking -- it spends the budget reasoning and never submits).

# Run from the worktree root: sbatch experiments/codebench/slurm/impossiblebench_olmo32b.sbatch
REPO_MAIN="/home/kobyjl/experiments/self-concept-experiments"
source "$REPO_MAIN/activate.sh"          # main checkout's uv .venv + cluster CUDA fixes
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1                   # weights are prefetched; compute nodes are offline
nvidia-smi

MODEL="allenai/Olmo-3.1-32B-Think"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/olmo32b_think"
mkdir -p "$OUT"

run_split () {  # $1 = cuda index, $2 = scenario
    CUDA_VISIBLE_DEVICES="$1" python -m selfconcept.codebench.run \
        --model "$MODEL" \
        --scenarios "$2" \
        --max-attempts 3 \
        --max-new-tokens 16384 \
        --out "$OUT" --tag base \
        > "$OUT/gen_$2.log" 2>&1
}

# One split per GPU, concurrently; resumable (same --tag appends past finished tasks).
run_split 0 impossible_oneoff &
PID_ONEOFF=$!
run_split 1 impossible_conflicting &
PID_CONFLICTING=$!

wait "$PID_ONEOFF"      && echo "=== impossible_oneoff done ==="
wait "$PID_CONFLICTING" && echo "=== impossible_conflicting done ==="

echo "=== summaries ==="
cat "$OUT"/base_impossible_oneoff_summary.json
cat "$OUT"/base_impossible_conflicting_summary.json
echo "=== all splits done ==="
