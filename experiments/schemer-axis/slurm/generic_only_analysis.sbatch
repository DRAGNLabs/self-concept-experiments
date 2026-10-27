#!/bin/bash --login
#SBATCH --job-name=schemer-generic-only
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis
set -euo pipefail

# Paper figures and numbers for the generic-only deceiver axis.  CPU only.  Projects the extension's response-only
# activations (100 questions) onto the canonical generic-only axis; per-role vectors are refit on the 40 generic
# questions for the persona geometry and the refit axis is compared with the canonical one.
SCRATCH="${SCRATCH:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis}"
EXTENSION_ROOT="${EXTENSION_ROOT:-$SCRATCH/gptoss120b_extension/full}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/gptoss120b_generic_only}"
CANONICAL_AXIS="${CANONICAL_AXIS:-$HOME/groups/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis/gptoss120b/roles_fic/candidates_response_only_17role/persona_contrast.pt}"
CANONICAL_VECTORS_DIR="${CANONICAL_VECTORS_DIR:-$(dirname "$(dirname "$CANONICAL_AXIS")")/vectors_response_only}"
MODEL=openai/gpt-oss-120b
LAYER=17

export OPENSSL_CONF=/dev/null
unset SELFCONCEPT_VENV
source ../../activate.sh

mkdir -p "$OUT_ROOT"
ln -sfn "$EXTENSION_ROOT/act_response_only" "$OUT_ROOT/act_response_only"
ln -sfn "$EXTENSION_ROOT/scores" "$OUT_ROOT/scores"

GENERIC_QUESTIONS="$(uv run python -c 'import json, sys; print([q["id"] for q in map(json.loads, open(sys.argv[1])) if q["group"] == "generic"])' data/questions.jsonl)"
(cd ../assistant-axis && uv run python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count 50 \
    --run.question_indices "$GENERIC_QUESTIONS" --run.activations_dir "$OUT_ROOT/act_response_only" --run.scores_dir "$OUT_ROOT/scores" \
    --run.output_dir "$OUT_ROOT/vectors_response_only")

read -r -a DECEPTIVE_ROLES <<< "$(uv run python -c 'import json; print(*json.load(open("data/persona_groups.json"))["deceptive_main"])')"
read -r -a HONEST_ROLES <<< "$(uv run python -c 'import json; print(*json.load(open("data/persona_groups.json"))["honest_main"])')"
uv run python scripts/build_vectors.py --vectors-dir "$OUT_ROOT/vectors_response_only" --model "$MODEL" --out "$OUT_ROOT/axis_response_only" \
    --schemer "${DECEPTIVE_ROLES[@]}" --honest "${HONEST_ROLES[@]}" --layers "$LAYER" > /dev/null

uv run python -c "
import sys, torch
canonical, refit = (torch.load(path, map_location='cpu', weights_only=False)['axis'].float() for path in sys.argv[1:3])
cosine = torch.nn.functional.cosine_similarity(canonical, refit, dim=-1)
print(f'cosine(canonical, refit from extension activations) layer $LAYER: {cosine[$LAYER].item():.4f}; all layers min {cosine.min().item():.4f}')
" "$CANONICAL_AXIS" "$OUT_ROOT/axis_response_only/persona_contrast.pt"

uv run python scripts/persona_figures.py --run.root "$OUT_ROOT" --run.region response_only --run.layer "$LAYER" \
    --run.axis "$CANONICAL_AXIS" --run.question_recipe generic_fit --run.model_label "GPT-OSS 120B" --run.out "$OUT_ROOT/figures_response_only"

for PERSONA_SET in heldout rh_adjacent; do
    uv run python scripts/persona_holdout_auroc.py --root "$OUT_ROOT" --axis-region response_only --score-region response_only \
        --persona-set "$PERSONA_SET" --layer "$LAYER" --fit-question-groups generic --eval-question-groups intent task \
        --out "$OUT_ROOT/analysis/persona_${PERSONA_SET}_generic_fit_L${LAYER}.json"
done

export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run python scripts/logit_lens.py --axis "$CANONICAL_AXIS" --vectors-dir "$CANONICAL_VECTORS_DIR" --layers "$LAYER" --out "$OUT_ROOT/logit_lens_persona_contrast"
uv run python scripts/logit_lens.py --axis "$OUT_ROOT/axis_response_only/persona_contrast_no_costume.pt" --layers "$LAYER" --out "$OUT_ROOT/logit_lens_persona_contrast_no_costume"
