#!/bin/bash --login
#SBATCH --job-name=schemer-cap-tau
#SBATCH --time=02:00:00
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

# Cap thresholds for the Phase 3 cap arm (PLAN.md): tau is a percentile of the per-token projection onto the
# direction in unsteered coding runs. The HF harness generates N unsteered episodes per direction with the
# projection sidecar on the band LAYERS (per-turn quantiles over the generated tokens); tau at each layer is the
# median over turns of the per-turn 50th (p50) or 25th (p25) percentile. Writes $ROOT/<name>_p50.json and _p25.json
# in the {layer: threshold} format steered_rh_vllm.py --cap-thresholds reads. Resumable (the harness skips finished ids).
# Usage: sbatch --export=ALL,DIRECTIONS="name=path ...",ROOT=<dir>[,N=4,MODEL=...,REVISION=...,CHAT_KWARGS='...',LAYERS="12 13 ... 23",TOPP=1.0,TOPK=0] experiments/schemer-axis/slurm/cap_thresholds.sbatch
# Defaults are the gpt-oss main run (low effort, band 12-23 around layer 17 of 36). Muse Glimmer (2026-10-21):
#   MODEL=meta-models/Muse-Glimmer-30B REVISION= CHAT_KWARGS='{"reasoning_strength":"low"}' LAYERS="17 ... 33" TOPP=0.95 TOPK=64
#   (the same third of the depth around layer 26 of 52; sampling as the Muse arms), on dwmatrix/eng/m13h with --gpus-per-node=1.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
DEFAULT_CHAT_KWARGS='{"reasoning_effort":"low"}'
export SOO_CHAT_KWARGS="${CHAT_KWARGS:-$DEFAULT_CHAT_KWARGS}"
# Scratch guard: Python's tempfile falls back to the working directory (this folder, via --chdir) when $TMPDIR and /tmp
# are unwritable, which left vllm_dist_* files, tmp* dirs and a torchinductor_* cache here after the 2026-10-20
# preempted-and-requeued pilot attempts on dw-2-5. Point TMPDIR at a per-job nobackup dir in that case.
export TMPDIR="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$HOME/nobackup/autodelete/tmp/slurm-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"; echo "TMPDIR fallback: $TMPDIR"
fi
nvidia-smi
# cs A100 nodes: a 1-GPU task can see a sibling's GPU; pin to the least-used one by UUID (see main_run.sbatch).
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"
if [ "$MODEL" = "openai/gpt-oss-120b" ]; then
    REVISION="${REVISION-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"  # the pinned gpt-oss snapshot unless overridden
else
    REVISION="${REVISION:-}"  # other models: the cached snapshot
fi
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"  # Koby's shared data (rh_hillclimb/data)
DIRECTIONS="${DIRECTIONS:?set DIRECTIONS to name=path pairs}"
ROOT="${ROOT:?set ROOT}"
N="${N:-4}"
LAYERS_STR="${LAYERS:-12 13 14 15 16 17 18 19 20 21 22 23}"
read -r -a LAYERS <<< "$LAYERS_STR"
TOPP="${TOPP:-1.0}"
TOPK="${TOPK:-0}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed 0
        --top-p "$TOPP" --top-k "$TOPK")
mkdir -p "$ROOT"
echo "model $MODEL revision '${REVISION}' chat kwargs $SOO_CHAT_KWARGS layers ${LAYERS[*]} top-p $TOPP top-k $TOPK n $N"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

for pair in $DIRECTIONS; do
    name="${pair%%=*}"; path="${pair#*=}"
    echo "########## projections: $name ($path) ##########"
    SECONDS=0
    python scripts/steered_rh.py --model "$MODEL" ${REVISION:+--revision "$REVISION"} --direction "$path" \
        --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" --projection-layers "${LAYERS[@]}" \
        --out "$ROOT/$name" --tag proj
    echo "$name episodes took ${SECONDS}s"
    python - "$ROOT" "$name" "${LAYERS[@]}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root, name = Path(sys.argv[1]), sys.argv[2]
band = [int(x) for x in sys.argv[3:]]
rows = [json.loads(l) for l in (root / name / "proj_ib_solvhard_s0_projection.jsonl").open()]
for q, label in (("0.5", "p50"), ("0.25", "p25")):
    tau = {}
    for layer in sorted({int(k) for r in rows for k in r["layers"]}):
        per_turn = [r["layers"][str(layer)]["generated_projection"]["quantiles"][q] for r in rows
                    if str(layer) in r["layers"] and r["layers"][str(layer)]["generated_projection"].get("n", 0) > 0]
        tau[str(layer)] = float(np.median(per_turn))
    (root / f"{name}_{label}.json").write_text(json.dumps(tau, indent=1))
    print(name, label, "turns", len(rows), {k: round(v, 1) for k, v in tau.items()})
show = (band[0], band[len(band) // 2], band[-1])
means = {str(l): round(float(np.mean([r["layers"][str(l)]["generated_projection"]["mean"] for r in rows])), 1) for l in show}
norms = {str(l): round(float(np.mean([r["layers"][str(l)]["generated_norm_mean"] for r in rows])), 1) for l in show}
print(name, "generated-token mean projection by layer", means, "mean token norm", norms)
PY
done
echo "=== schemer-cap-tau done: $ROOT ==="
