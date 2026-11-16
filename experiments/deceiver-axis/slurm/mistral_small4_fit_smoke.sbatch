#!/bin/bash --login
#SBATCH --job-name=ms4-fit-smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=dw87
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# One-GPU fit check for mistralai/Mistral-Small-4-119B-2603 before a Stage C run.
# The native FP8 shards are about 121 GB. A B200 has 180 GB, so one card holds the
# weights plus a 32k context. The two-GPU recipe is the 256k serving setup.
# The job expects the consolidated shards already in HF_HOME. It does not download.
# Job 15317405 loaded the weights, then died because FLASHINFER_MLA tried to
# download a cubin. The smoke forces TRITON_MLA, which is already installed.
# Job 15318686 loaded and generated one empty token because the prompt was the
# chat template rendered to text and encoded again. The prompt is now the
# template's token ids.
#
# gpu_pick 2026-11-12, --hours 2 --vram-gb 160 --cpus 8 --mem-gb 128 --cuda 13.0
# --no-requeue-ok, smaller GPUs excluded: the 160 GB total can be two 80 GB cards,
# and those cannot hold this checkpoint. The remaining row is one B200 on cs3 / cs
# (16 jobs ahead, none idle). The default account nturley does not include qos cs;
# account dw87 does. Saved with the run under $ROOT/gpu_pick.txt.
#
# Usage, from the repo root, once the consolidated shards are in HF_HOME:
#   sbatch experiments/deceiver-axis/slurm/mistral_small4_fit_smoke.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
nvidia-smi

ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/mistral-small4/fit_smoke}"
mkdir -p "$ROOT"

source ../../activate.sh
python scripts/mistral_small4_fit_smoke.py --out "$ROOT" --max-model-len "${MAX_MODEL_LEN:-32768}"
echo "=== ms4 fit smoke done: $ROOT ==="
