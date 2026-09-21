#!/bin/bash --login
#SBATCH --job-name=soo2-subspace-qwen38
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=experiments/soo/slurm-logs/%x-%j.out

# Submit from the repository root, passing an absolute frozen snapshot path.
# Keep the GPU assigned by Slurm; never borrow another job's GPU.
set -euo pipefail
SNAPSHOT_ROOT=${1:?pass the frozen snapshot directory}
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit through Slurm from the repository root}
export PYTHONPATH="$SNAPSHOT_ROOT/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOO_CHAT_KWARGS='{"enable_thinking": false}'
export OMP_NUM_THREADS=4
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm did not assign a visible GPU; refusing to select an unallocated GPU."
    exit 1
fi
echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd "$SNAPSHOT_ROOT"
"$REPO_ROOT/.venv/bin/python" -u -m selfconcept.soo.measure_overlap \
    --model Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --probes data/subspace_pilot_pairs.jsonl --split development \
    --vectors data/qwen38_27b.pt --mode add --alpha 10 --token-mode last \
    --layer 31 --residual-layers 31 32 39 47 55 63 --positions all \
    --subspace-ranks 1 2 4 8 --subspace-strengths 0 0.5 1 --random-seeds 0 1 2 \
    --batch-size 4 --device cuda --dtype bfloat16 --bootstrap 2000 --seed 0 \
    --checkpoint-conditions --out output
