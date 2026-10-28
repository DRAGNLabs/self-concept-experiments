#!/bin/bash
# Activate the uv venv and apply the cluster-specific runtime fixes that used to
# live in conda activate.d hooks (before the mamba->uv migration). Source this,
# don't execute it:
#   source /path/to/repo/activate.sh
# From a slurm script chdir'd into experiments/<x>, that's:
#   source ../../activate.sh
# For the P100/V100 nodes (cu126 torch), pass p100:
#   source activate.sh p100
# after building that venv once with:
#   UV_PROJECT_ENVIRONMENT=.venv-p100 uv sync --no-default-groups --group dev --group p100
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "p100" ]; then
    source "$HERE/.venv-p100/bin/activate"
    unset CUDA_HOME FLASHINFER_WORKSPACE_BASE
else
    # SELFCONCEPT_VENV overrides the venv directory (e.g. a .venv-313 built next to a stale .venv).
    source "${SELFCONCEPT_VENV:-$HERE/.venv}/bin/activate"

    # flashinfer JIT-compiles kernels (e.g. vllm's default attention on B200) with
    # the venv's own nvcc, but expects a standard CUDA_HOME layout: bin/, include/
    # and an unversioned lib64/libcudart.so, none of which the pip wheels provide
    # in that shape. Assemble one from symlinks. libcuda.so (the driver) comes from
    # the node.
    _cu13_dir="$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13"
    export CUDA_HOME="$VIRTUAL_ENV/cuda-home"
    mkdir -p "$CUDA_HOME/lib64"
    for _cuda_subdir in bin include nvvm; do
        ln -sfn "$_cu13_dir/$_cuda_subdir" "$CUDA_HOME/$_cuda_subdir"
    done
    ln -sfn "$_cu13_dir/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
    unset _cu13_dir _cuda_subdir
    export FLASHINFER_WORKSPACE_BASE="$HOME/nobackup/autodelete/self-concept-experiments/flashinfer"
fi

# vllm's flashinfer sampler JIT-compiles with nvcc, which only the default venv
# has set up (CUDA_HOME above); the p100 venv has none. Kept off in both so
# sampling uses the same (Triton) path as earlier runs.
export VLLM_USE_FLASHINFER_SAMPLER=0

# vllm's cu13 kernels dlopen libcudart.so.13, which pip ships under the venv's
# nvidia/cu13/lib but leaves off every loader path. Without this it resolves only
# on nodes whose OS image registers a system CUDA-13 (login, some A100s) and fails
# on nodes registering only CUDA-12 (H200 partitions m13h/eng). Point the loader
# at the venv's own cu13 libdir. Safe alongside torch's cu12 libs: cuda sonames
# are major-versioned (libcudart.so.12 vs .13), so a cu12 consumer never picks
# these up. Guarded so it's a no-op if the resolved vllm build ships no cu13 libs.
_cu13_libdir="$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13/lib"
if [ -d "$_cu13_libdir" ]; then
    export LD_LIBRARY_PATH="$_cu13_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _cu13_libdir
