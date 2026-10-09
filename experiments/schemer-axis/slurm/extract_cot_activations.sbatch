#!/bin/bash --login
#SBATCH --job-name=schemer-cot-act
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Re-run only the CoT-region activation pass (2_activations with include_cot=true) and its vectors for an existing
# extraction root. Until 2026-10-09 the gpt-oss model specifics ignored enable_thinking and pooled the final channel,
# so every earlier act_with_cot/ was a copy of act_response_only/ (FINDINGS, Phase 1). The target directories must
# not exist (2_activations skips roles whose .pt file is present).
# Usage: sbatch --export=ALL,ROOT=<extraction root>[,ROLES="..."] experiments/schemer-axis/slurm/extract_cot_activations.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
nvidia-smi

MODEL="openai/gpt-oss-120b"
MAXLEN=20480
ACT_BATCH=4
ATTENTION="kernels-community/vllm-flash-attn3"
MIN_COUNT="${MIN_COUNT:-50}"
ROLES="${ROLES:-default schemer_fic deceiver_fic cheater_fic straight_shooter_fic rule_follower_fic neutral_fic}"
ROLES_BRACKET="[$(echo $ROLES | tr ' ' ',')]"
ROOT="${ROOT:?set ROOT to the extraction root (holds responses/ and scores/)}"
for d in act_with_cot vectors_with_cot; do
    if [ -e "$ROOT/$d" ]; then echo "refusing to run: $ROOT/$d exists (remove it first)"; exit 1; fi
done

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

echo "########## Extract (with_cot, include_cot=true) ##########"
python scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml \
    --run.model "$MODEL" --run.include_cot true --run.max_length "$MAXLEN" --run.batch_size "$ACT_BATCH" \
    --run.attn_implementation "$ATTENTION" --run.roles "$ROLES_BRACKET" \
    --run.responses_dir "$ROOT/responses" --run.output_dir "$ROOT/act_with_cot"
python scripts/pipeline/4_vectors.py --config configs/pipeline/4_vectors.yaml --run.min_count "$MIN_COUNT" \
    --run.activations_dir "$ROOT/act_with_cot" --run.scores_dir "$ROOT/scores" --run.output_dir "$ROOT/vectors_with_cot"

python - "$ROOT" <<'PY'
import sys, torch
from pathlib import Path
root = Path(sys.argv[1])
for role in ["schemer_fic", "default"]:
    a = torch.load(root / "act_response_only" / f"{role}.pt", map_location="cpu", weights_only=False)
    b = torch.load(root / "act_with_cot" / f"{role}.pt", map_location="cpu", weights_only=False)
    keys = sorted(set(a) & set(b))
    cos = torch.nn.functional.cosine_similarity(torch.stack([a[k][17] for k in keys]), torch.stack([b[k][17] for k in keys]), dim=-1)
    print(f"{role}: {len(a)} final-region vs {len(b)} CoT-region activations; layer-17 cosine final vs CoT per response: "
          f"mean {cos.mean():.3f}, min {cos.min():.3f}, max {cos.max():.3f}")
PY
echo "=== schemer-cot-act done: $ROOT ==="
