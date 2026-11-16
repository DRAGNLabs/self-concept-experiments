#!/bin/bash --login
#SBATCH --job-name=beam-nvfp4-smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=dw87
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:2
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Two-B200 fit check for reflection/Beam-1.0-501B-A23B-NVFP4.
# The NVFP4 checkpoint is about 274 GiB. vLLM's published minimum is 353 GB
# of GPU memory, so one B200 cannot hold it and two can. The published
# comfortable layout is four B200s; this smoke is the two-card test.
# Reasoning effort is minimal (budget 8). The prompt is the template's token ids.
#
# The NVFP4 snapshot is in HF_HOME. This job uses .venv-beam (vLLM 0.33.0),
# which imports BeamForCausalLM. The shared .venv stays on vLLM 0.28.
# Do not submit the 5-seed collection from this script.
#
# Recheck gpu_pick.py at submission time (--hours 2 --vram-gb 320 --cpus 16
# --mem-gb 256 --cuda 13.0 --no-requeue-ok, and exclude GPUs that are not
# Blackwell). NVFP4 needs B200s. Save that output next to the run.
#
# Usage, from the repo root, after the checks above:
#   sbatch experiments/deceiver-axis/slurm/beam_nvfp4_fit_smoke.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_ENGINE_READY_TIMEOUT_S=3600
nvidia-smi

ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/beam-nvfp4/fit_smoke}"
mkdir -p "$ROOT"

export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$(cd ../.. && pwd)/.venv-beam}"
source ../../activate.sh
python scripts/beam_nvfp4_fit_smoke.py --out "$ROOT" --max-model-len "${MAX_MODEL_LEN:-8192}"
echo "=== beam nvfp4 fit smoke done: $ROOT ==="
