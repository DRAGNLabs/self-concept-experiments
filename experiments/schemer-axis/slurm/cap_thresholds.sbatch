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

# Cap thresholds for the Phase 3 dose pilot (PLAN.md): tau is a percentile of the per-token projection onto the
# direction in unsteered coding runs. The HF harness generates N unsteered episodes per direction with the
# projection sidecar on layers 12-23 (per-turn quantiles over the generated tokens); tau at each layer is the
# median over turns of the per-turn 50th (p50) or 25th (p25) percentile. Writes $ROOT/<name>_p50.json and _p25.json
# in the {layer: threshold} format steered_rh_vllm.py --cap-thresholds reads.
# Usage: sbatch --export=ALL,DIRECTIONS="name=path ...",ROOT=<dir>[,N=4] experiments/schemer-axis/slurm/cap_thresholds.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
nvidia-smi

MODEL="openai/gpt-oss-120b"
REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
DIRECTIONS="${DIRECTIONS:?set DIRECTIONS to name=path pairs}"
ROOT="${ROOT:?set ROOT}"
N="${N:-4}"
LAYERS=(12 13 14 15 16 17 18 19 20 21 22 23)
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed 0)
mkdir -p "$ROOT"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

for pair in $DIRECTIONS; do
    name="${pair%%=*}"; path="${pair#*=}"
    echo "########## projections: $name ($path) ##########"
    python scripts/steered_rh.py --model "$MODEL" --revision "$REVISION" --direction "$path" \
        --scenarios ib_solvhard_s0 --n "$N" "${RECIPE[@]}" --projection-layers "${LAYERS[@]}" \
        --out "$ROOT/$name" --tag proj
    python - "$ROOT" "$name" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root, name = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in (root / name / "proj_ib_solvhard_s0_projection.jsonl").open()]
for q, label in (("0.5", "p50"), ("0.25", "p25")):
    tau = {}
    for layer in sorted({int(k) for r in rows for k in r["layers"]}):
        per_turn = [r["layers"][str(layer)]["generated_projection"]["quantiles"][q] for r in rows
                    if str(layer) in r["layers"] and r["layers"][str(layer)]["generated_projection"].get("n", 0) > 0]
        tau[str(layer)] = float(np.median(per_turn))
    (root / f"{name}_{label}.json").write_text(json.dumps(tau, indent=1))
    print(name, label, "turns", len(rows), {k: round(v, 1) for k, v in tau.items()})
means = {str(l): round(float(np.mean([r["layers"][str(l)]["generated_projection"]["mean"] for r in rows])), 1) for l in (12, 17, 23)}
print(name, "generated-token mean projection by layer", means)
PY
done
echo "=== schemer-cap-tau done: $ROOT ==="
