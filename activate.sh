#!/bin/bash
# Activate the uv venv and apply the cluster-specific runtime fixes that used to
# live in conda activate.d hooks (before the mamba->uv migration). Source this,
# don't execute it:
#   source /path/to/repo/activate.sh
# From a slurm script chdir'd into experiments/<x>, that's:
#   source ../../activate.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"

# vllm's flashinfer sampler JIT-compiles a CUDA kernel by shelling out to nvcc
# (the CUDA *compiler*, which torch's runtime-only CUDA wheels don't bundle and
# this cluster has no toolkit for). The Triton fallback bundles its own compiler
# backend, so disable flashinfer's path.
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
