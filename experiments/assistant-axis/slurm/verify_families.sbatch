#!/bin/bash --login
#SBATCH --job-name=axis-verify
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Throwaway check that the pipeline still generates+extracts across model families after
# the step-1 `thinking` override and the step-2 `thinking`->`include_cot` rename (plus the
# get_model_specifics_by_name refactor). Isolated scratch subtrees; safe to delete after.
# Lane A (Qwen2.5-0.5B full pipeline) already passed in a prior run; this covers:
#   Lane B  gemma-2-2b-it  non-Qwen family    -> full 5 stages (regression)
#   Lane C  Qwen3-1.7B     thinking + CoT ON  -> stages 1-2 (the new paths)

mamba activate ../../.env
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"  # shared cache (offline on compute nodes)
nvidia-smi

ROLES="[default,accountant,comedian]"
ROOT="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/verify"
rm -rf "$ROOT"

echo "########## Lane B: gemma-2-2b-it (non-Qwen), full 5 stages ##########"
BSM="$ROOT/gemma"
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "google/gemma-2-2b-it" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count 8 --run.roles "$ROLES" --run.output_dir "$BSM/responses"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "google/gemma-2-2b-it" --run.layers 0,1 --run.roles "$ROLES" \
    --run.responses_dir "$BSM/responses" --run.output_dir "$BSM/activations"
python pipeline/3_judge.py --config configs/3_judge.yaml \
    --run.judge_model "google/gemma-2-2b-it" --run.tensor_parallel_size 1 --run.roles "$ROLES" \
    --run.responses_dir "$BSM/responses" --run.output_dir "$BSM/scores"
python pipeline/4_vectors.py --config configs/4_vectors.yaml --run.min_count 1 \
    --run.activations_dir "$BSM/activations" --run.scores_dir "$BSM/scores" \
    --run.output_dir "$BSM/vectors"
# A tiny 2b judge often gives 0 score=3 for the contrast roles, leaving no contrast
# vectors -> step 5 exits 1 by design. Don't let that expected data-shortfall abort Lane C
# (step 5's happy path already passed in the Qwen lane).
python pipeline/5_axis.py --config configs/5_axis.yaml \
    --run.vectors_dir "$BSM/vectors" --run.output "$BSM/axis.pt" \
    || echo "Lane B: step 5 produced no axis (expected — weak 2b judge, no contrast vectors)"

echo "########## Lane C: Qwen3-1.7B (thinking + include_cot ON) ##########"
CSM="$ROOT/qwen3"
python pipeline/1_generate.py --config configs/1_generate.yaml \
    --run.model "Qwen/Qwen3-1.7B" --run.tensor_parallel_size 1 --run.gpu_memory_utilization 0.9 \
    --run.question_count 8 --run.max_tokens 1024 --run.thinking true \
    --run.roles "$ROLES" --run.output_dir "$CSM/responses"
python pipeline/2_activations.py --config configs/2_activations.yaml \
    --run.model "Qwen/Qwen3-1.7B" --run.layers 0,1 --run.include_cot true --run.roles "$ROLES" \
    --run.responses_dir "$CSM/responses" --run.output_dir "$CSM/activations"

echo "########## Verification summary ##########"
python - "$ROOT" <<'PY'
import sys, pathlib, json
root = pathlib.Path(sys.argv[1])
def n_pt(p): return len(list((root / p).glob("*.pt"))) if (root / p).exists() else 0
# Lane B: full pipeline for a non-Qwen family
print(f"[B gemma-2-2b] activations={n_pt('gemma/activations')} vectors={n_pt('gemma/vectors')} "
      f"axis={'yes' if (root/'gemma/axis.pt').exists() else 'NO'}")
# Lane C: prove thinking generation happened and extraction ran with include_cot
c = root / "qwen3"
resp = sorted((c / "responses").glob("*.jsonl"))
has_cot = any(
    "</think>" in (m.get("content") or "")
    for f in resp for line in f.open()
    for m in json.loads(line)["conversation"]
)
print(f"[C qwen3-1.7b thinking+cot] responses={len(resp)} contains_</think>={has_cot} "
      f"activations={n_pt('qwen3/activations')}")
PY

echo "=== verify_families done ==="
