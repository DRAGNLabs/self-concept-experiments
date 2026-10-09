#!/bin/bash --login
#SBATCH --job-name=schemer-vllm-logprob
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Phase 3 backend check, part 3: is the steering active inside the compiled engine, at the right magnitude?
# Greedy text cannot show it (eager and compiled decodes diverge on bf16 noise). Instead, fixed texts are scored
# (prompt log-probabilities per token) under six engines: {eager, compiled} x {unsteered, add +0.4, cap tau=300}.
# Expected: eager-vs-compiled differences within an arm are at the eager-unsteered-vs-compiled-unsteered noise floor,
# and steered-vs-unsteered differences are far above it, with the same magnitude in eager and compiled mode.
# Usage: sbatch experiments/schemer-axis/slurm/steered_rh_vllm_logprob_check.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1  # cached graphs are keyed on the model, not on the steering patch
nvidia-smi

MODEL="openai/gpt-oss-120b"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
DIRECTION="${DIRECTION:-$MIRROR/assistant-axis/gptoss12b-thinking/axis_response_only.py}"
SMOKE="$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/vllm_compiled_smoke"
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/vllm_logprob_check}"
NORMS="${NORMS:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/rh_smoke/norms/layer_norms.pt}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --sample-seed 0)
mkdir -p "$ROOT"
echo '{"17": 300.0}' > "$ROOT/cap_thresholds.json"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

# Fixed texts: the three unsteered and the three compiled-add greedy answers from the compiled smoke.
python - "$SMOKE" "$ROOT/texts.jsonl" <<'PY'
import json, sys
from pathlib import Path
smoke, out = Path(sys.argv[1]), Path(sys.argv[2])
with out.open("w") as f:
    for d in ["probe_unsteered", "probe_compiled"]:
        for line in (smoke / d / "probe.jsonl").open():
            r = json.loads(line)
            f.write(json.dumps({"question": r["question"], "final": r["final"], "source": d}) + "\n")
print("texts:", sum(1 for _ in out.open()))
PY

ADD=(--direction "$DIRECTION" --layer-norms "$NORMS" --steer-mode add --steer-layer 17 --steer-alpha 0.4)
CAP=(--direction "$DIRECTION" --steer-mode cap --cap-layers 17 --cap-thresholds "$ROOT/cap_thresholds.json")
run() {  # name, then extra args
    local name=$1; shift
    echo "########## score: $name ##########"
    python scripts/steered_rh_vllm.py --model "$MODEL" --temperature 0 --scenarios ib_solvhard_s0 "${RECIPE[@]}" \
        --out "$ROOT/$name" --tag score --score "$ROOT/texts.jsonl" "$@"
}
run unsteered_eager --eager
run unsteered_compiled
run add_eager --eager "${ADD[@]}"
run add_compiled "${ADD[@]}"
run cap_eager --eager "${CAP[@]}"
run cap_compiled "${CAP[@]}"

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
names = ["unsteered_eager", "unsteered_compiled", "add_eager", "add_compiled", "cap_eager", "cap_compiled"]
rows = {n: [json.loads(l) for l in (root / n / "score.jsonl").open()] for n in names}
def diff(a, b):
    out = []
    for ra, rb in zip(rows[a], rows[b]):
        assert ra["text_tokens"] == rb["text_tokens"]
        d = [abs(x - y) for x, y in zip(ra["logprobs"], rb["logprobs"])]
        out.append((sum(d) / len(d), max(d), ra["sum_logprob"] - rb["sum_logprob"]))
    return out
pairs = [("unsteered_eager", "unsteered_compiled", "noise floor"), ("add_eager", "add_compiled", "add: eager vs compiled"),
         ("cap_eager", "cap_compiled", "cap: eager vs compiled"), ("add_eager", "unsteered_eager", "add effect, eager"),
         ("add_compiled", "unsteered_compiled", "add effect, compiled"), ("cap_eager", "unsteered_eager", "cap effect, eager"),
         ("cap_compiled", "unsteered_compiled", "cap effect, compiled")]
print(f"{'pair':28s} " + " ".join(f"text{i}: mean|d| max|d| dsum" for i in range(len(rows['unsteered_eager']))))
for a, b, label in pairs:
    cells = [f"{m:.4f} {mx:.3f} {ds:+.1f}" for m, mx, ds in diff(a, b)]
    print(f"{label:28s} " + " | ".join(cells))
summary = {label: diff(a, b) for a, b, label in pairs}
(root / "summary.json").write_text(json.dumps(summary, indent=1))
PY
echo "=== schemer-vllm-logprob done: $ROOT ==="
