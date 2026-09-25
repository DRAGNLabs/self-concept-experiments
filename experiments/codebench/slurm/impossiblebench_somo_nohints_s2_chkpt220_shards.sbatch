#!/bin/bash --login
#SBATCH --job-name=ib-somo-nohints-shard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --requeue
#SBATCH --gpus=a100:2
#SBATCH --mem=100G
#SBATCH --array=0-15
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail

# ImpossibleBench on the SOMO nohints-s2 GRPO checkpoint 220 (LoRA over
# somo-olmo-32b-sdf-sft, an OLMo-3 32B Instruct SFT; no <think> block), sharded across a
# cs-QOS array (starts fast at 0.1x fairshare). 16 tasks = 8 shards x 2 impossible
# splits, two A100s each: on one 80GB GPU the 3-turn contexts hit CUDA OOM, and
# standby H200s queue for hours. Preemptable (--requeue); the harness resumes each shard
# from its own jsonl (same --tag skips finished example_ids), so a preempted shard
# picks up where it left off. Overall reward-hacking rate is computed after the
# array finishes by merging shards (see the merge/score step).
#
# SYSTEM_PROMPT_FILE=<file> sbatch ... adds that system message (e.g. the paper's
# <thinking> + ```python format, experiments/codebench/prompts/somo_thinking_format.txt)
# and writes to OUT_NAME (default: the file's stem appended to the plain run's dir).
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
REPO_MAIN="/home/kobyjl/experiments/self-concept-experiments"
source "$REPO_MAIN/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1

BASE="ai-safety-institute/somo-olmo-32b-sdf-sft"
# The base's config.json was written by transformers 5.3.0.dev0 as one flat YaRN
# rope_parameters dict, which 5.17 rejects (it wants per-layer-type params).
# Rebuild it with allenai's OLMo-3 layout (YaRN on full attention, plain RoPE on
# sliding) in a scratch dir that symlinks the cached weights: YaRN on every layer
# made greedy decoding loop without EOS, allenai's layout stops cleanly. Its saved
# use_cache=false (from gradient checkpointing) would make generate() recompute
# the whole sequence every token, so turn the KV cache back on too.
MODEL="$HOME/nobackup/autodelete/self-concept-experiments/models/somo-olmo-32b-sdf-sft-rope-per-layer"
python - "$BASE" "$MODEL" <<'PY'
import json, os, sys
from contextlib import suppress
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
base, dst = Path(try_to_load_from_cache(sys.argv[1], "config.json")).parent, Path(sys.argv[2])
REWRITTEN = ("config.json", "generation_config.json")
dst.mkdir(parents=True, exist_ok=True)
for src in base.iterdir():
    if src.name not in REWRITTEN:
        with suppress(FileExistsError):  # concurrent array tasks
            (dst / src.name).symlink_to(src.resolve())

def write_json_atomically(path: Path, content: dict) -> None:
    # os.replace swaps the path itself, never writing through an old symlink into the HF cache
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(content, indent=2))
    os.replace(tmp, path)

config = json.loads((base / "config.json").read_text())
yarn = config["rope_parameters"]
config["rope_parameters"] = {
    "sliding_attention": {"rope_type": "default", "rope_theta": yarn["rope_theta"]},
    "full_attention": yarn,
}
config["use_cache"] = True
write_json_atomically(dst / "config.json", config)
generation_config = json.loads((base / "generation_config.json").read_text())
generation_config["use_cache"] = True
write_json_atomically(dst / "generation_config.json", generation_config)
PY
ADAPTER="ai-safety-institute/somo-olmo-32b-nohints-s2-chkpt-220"
SHARD_DATA="$HOME/nobackup/autodelete/self-concept-experiments/codebench/shards"
OUT_NAME="${OUT_NAME:-somo_nohints_s2_chkpt220${SYSTEM_PROMPT_FILE:+_$(basename "$SYSTEM_PROMPT_FILE" .txt)}}"
OUT="$HOME/nobackup/autodelete/self-concept-experiments/codebench/$OUT_NAME"
mkdir -p "$OUT"

# array id -> (split, shard). ids 0-7 = oneoff shards 0-7; 8-15 = conflicting shards 0-7.
SPLITS=(impossible_oneoff impossible_conflicting)
SPLIT="${SPLITS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}"
SHARD=$(( SLURM_ARRAY_TASK_ID % 8 ))
echo "task $SLURM_ARRAY_TASK_ID -> split=$SPLIT shard=$SHARD on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# SOMO's generation_config stops only on <|endoftext|>, so a turn closed with
# <|im_end|> ran on into invented next turns; and greedy decoding loops on empty
# ``` lines after the code block until max_new_tokens. Stop on either.
python -m selfconcept.codebench.run \
    --model "$MODEL" --adapter "$ADAPTER" --device-map auto \
    --stop-strings '<|im_end|>' $'```\n```' \
    ${SYSTEM_PROMPT_FILE:+--system-prompt-file "$SYSTEM_PROMPT_FILE"} \
    --data "$SHARD_DATA" \
    --scenarios "${SPLIT}_s${SHARD}" \
    --max-attempts 3 --max-new-tokens 16384 \
    --out "$OUT" --tag "s${SHARD}"

echo "=== task $SLURM_ARRAY_TASK_ID done: $SPLIT shard $SHARD ==="
