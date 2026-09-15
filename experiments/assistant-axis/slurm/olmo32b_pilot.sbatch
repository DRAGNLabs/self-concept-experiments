#!/bin/bash --login
#SBATCH --job-name=axis-olmo32b-pilot
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Timing pilot for the OLMo-3.1-32B-Think full-scale sweep (17 roles x 40 questions,
# thinking on, max_tokens=32768 per the model card's recommended budget). Single GPU
# only: [[vllm028-multigpu-node-startup-hang]] means the ported env's vLLM 0.28 hangs
# at engine startup on any >1-GPU node allocation, so TP>1 isn't an option here. A 32B
# model in bf16 (~65GB) fits comfortably on one H200 (141GB) with headroom for KV cache.
# Generates a small batch (3 roles x 8 questions = 24 prompts) at the real token budget
# and reports wall time + how many responses actually closed their <think> block, to
# extrapolate cost/feasibility before committing to the full 680-prompt x 2-config run.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="allenai/Olmo-3.1-32B-Think"
ROLES="[default,accountant,comedian]"
QUESTIONS=8
MAXTOK=32768
MAXLEN=40960
PILOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/olmo32b-pilot"
rm -rf "$PILOT"

mamba activate ../../.env

echo "########## Generate WITH thinking, max_tokens=$MAXTOK ##########"
start=$SECONDS
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "$MODEL" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count "$QUESTIONS" --run.roles "$ROLES" \
    --run.thinking true --run.temperature 0.6 --run.top_p 0.95 \
    --run.max_tokens "$MAXTOK" --run.max_model_len "$MAXLEN" \
    --run.output_dir "$PILOT/responses"
elapsed=$((SECONDS - start))
echo "[TIMING] generation: ${elapsed}s for 24 prompts"

echo "########## Completion summary ##########"
python3 - "$PILOT/responses" <<'PY'
import sys, glob, json

root = sys.argv[1]
n = 0
closed = 0
lengths = []
for path in glob.glob(f"{root}/*.jsonl"):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            content = d["conversation"][-1]["content"]
            n += 1
            lengths.append(len(content))
            if "</think>" in content:
                closed += 1

lengths.sort()
print(f"responses={n} closed_think_block={closed}/{n}")
if lengths:
    print(f"content chars: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}")
PY

echo "=== olmo32b_pilot done ==="
