#!/bin/bash --login
#SBATCH --job-name=gi-v2-validation
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --no-requeue
#SBATCH --output=experiments/deceiver-axis/slurm-logs/%x-%j.out
set -euo pipefail

# The gi_v2 axis validation run for gpt-oss-120b (gptoss120b_v2/full), repeated for another model: per-token held-out
# projections and the token-extremes page, the persona figures (distributions with and without controls, PC1, centered
# cosine, plus the geometry numbers in the summary JSON) and the logit lens of the axis, all at the steering layer on
# the response_only region. ROOT is an extraction root holding every role in data/persona_groups.json.
# Usage (from the repo root): sbatch -p ... --gpus ... --export=ALL,MODEL=<hf id>,LAYER=<n>,ROOT=<dir>,AXIS=<persona_contrast.pt>,LABEL='<title>',OUT=<dir>[,TOKEN_BATCH=4] experiments/deceiver-axis/slurm/gi_v2_validation.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENSSL_CONF=/dev/null
: "${MODEL:?}" "${LAYER:?}" "${ROOT:?}" "${AXIS:?}" "${LABEL:?}" "${OUT:?}"
TOKEN_BATCH="${TOKEN_BATCH:-4}"
MAXLEN="${MAXLEN:-20480}"
source activate.sh
export PYTHONPATH="$PWD/src"
nvidia-smi
cd experiments/deceiver-axis

TOKEN_ROLES="default $(python -c 'import json; g=json.load(open("data/persona_groups.json")); print(" ".join(r for k in ("deceptive_holdout","honest_holdout","deceptive_rh","honest_rh","controls") for r in g[k]))')"
echo "########## token projections (L$LAYER): $TOKEN_ROLES ##########"
python scripts/token_projections.py --root "$ROOT" --response-only-axis "$AXIS" --layer "$LAYER" --model "$MODEL" --out "$OUT/tokens_L$LAYER" \
    --roles $TOKEN_ROLES --max-length "$MAXLEN" --batch-size "$TOKEN_BATCH"
python scripts/token_figures.py --run.tokens "$OUT/tokens_L$LAYER" --run.model "$MODEL" --run.region response_only --run.model_label "$LABEL" \
    --run.exclude_axis_personas "$AXIS" --run.out "$OUT/figures_response_only/response_only_L${LAYER}_token_extremes_heldout_personas.html"

echo "########## persona figures ##########"
python scripts/persona_figures.py --run.root "$ROOT" --run.region response_only --run.layer "$LAYER" --run.axis "$AXIS" \
    --run.model_label "$LABEL" --run.out "$OUT/figures_response_only"

echo "########## logit lens ##########"
python scripts/logit_lens.py --direction "persona_contrast=$AXIS" --model "$MODEL" --layers "$LAYER" --out "$OUT/logit_lens_response_only"
echo "=== gi-v2-validation done: $OUT ==="
