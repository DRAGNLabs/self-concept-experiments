#!/bin/bash --login
#SBATCH --job-name=axis-steered-smoke
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

# Step-6 plumbing check on a tiny model, one GPU: load the model, apply the steering hook,
# generate baseline + steered traces, and write the role_susceptibility_traces.md schema.
# Axis/norm quality is out of scope here (smoke.sbatch validates real axis production and
# activation_norms.py validates real norm measurement), so this fabricates a full-layer
# random axis and a flat norms tensor sized to the model. Writes to a smoke scratch subtree.

source ../../activate.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
nvidia-smi

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SMOKE="$HOME/nobackup/autodelete/self-concept-experiments/assistant-axis/smoke/steered"
mkdir -p "$SMOKE"

python - "$MODEL" "$SMOKE" <<'PY'
import sys
import torch
from transformers import AutoConfig

model, smoke = sys.argv[1], sys.argv[2]
config = AutoConfig.from_pretrained(model)
torch.manual_seed(0)
torch.save(torch.randn(config.num_hidden_layers, config.hidden_size), f"{smoke}/axis.pt")
torch.save(torch.full((config.num_hidden_layers,), 20.0), f"{smoke}/layer_norms.pt")
print(f"fabricated axis ({config.num_hidden_layers}, {config.hidden_size}) and norms for {model}")
PY

python scripts/steering/6_steered_traces.py --config configs/steering/6_steered_traces.yaml \
    --run.model "$MODEL" \
    --run.axis_path "$SMOKE/axis.pt" --run.norms_path "$SMOKE/layer_norms.pt" \
    --run.system_prompts_file data/steer_prompts.jsonl \
    --run.output_dir "$SMOKE/out" \
    --run.coefficients '[4.0]' --run.max_new_tokens 32

echo "=== Steered smoke done: $SMOKE/out ==="
