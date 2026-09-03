#!/bin/bash
# Applies environment-specific fixes to an already-existing ./.env. Called by both
# setup_env.sh (after `mamba env create`) and update_env.sh (after `mamba env
# update`, which re-runs the pip section and would otherwise reintroduce the
# opencv issue below). Idempotent -- safe to rerun on its own too.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# vllm pulls in opencv-python-headless transitively (for its optional video-input
# support only), whose bundled OpenSSL fails its own FIPS self-test on this cluster --
# crashes with SIGABRT on `import cv2`, and so on `from vllm import LLM` -- confirmed
# to reproduce on every opencv release from
# 4.13 through 5.0 tried. We never touch vllm's video features (text-only pipeline),
# and vllm's own `vllm/multimodal/video.py` already falls back cleanly to a
# PlaceholderModule when cv2 is simply *absent* (an ImportError it explicitly
# catches) -- it just can't survive cv2 being *present but crashing on load* (a
# SIGABRT, not a catchable exception). So the fix is to not install opencv at all,
# rather than to pin a version -- confirmed `from vllm import LLM` imports cleanly
# with it fully uninstalled.
# If/Once we switch to uv, they have a `[tool.uv] override-dependencies` option we
# can use to deal with this.
./.env/bin/pip uninstall -y opencv-python-headless

# Auto-export runtime fixes whenever this env is activated (`mamba activate ./.env`),
# so slurm scripts and interactive sessions don't need to repeat them.
mkdir -p ./.env/etc/conda/activate.d ./.env/etc/conda/deactivate.d

cat > ./.env/etc/conda/activate.d/vllm_env_fixes.sh <<'EOF'
# nvidia-cudnn-cu12 (a pip wheel torch depends on) needs libstdc++.so.6, but its
# own RUNPATH only covers its own package dir, so that lookup falls through to
# the system's older /usr/lib64/libstdc++.so.6 -- which then gets cached as THE
# libstdc++.so.6 for the rest of the process (a given soname loads once per
# process). Later, conda-forge's icu package (pulled in transitively by vllm,
# compiled with a newer GCC) needs a symbol only newer libstdc++ builds have
# (CXXABI_1.3.15) and crashes, stuck with the older copy cuDNN already loaded.
# Force our own (newer) copy to win that race instead.
# This is conda-specific: icu here is a conda-forge package, so it wouldn't
# exist at all in a uv/plain-venv setup -- the planned uv migration will likely
# make this fix unnecessary (verify with a plain `from vllm import LLM` once
# migrated, don't just assume).
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

# vllm's flashinfer sampler JIT-compiles a CUDA kernel on first use by shelling out
# to nvcc (the CUDA *compiler*, not just runtime -- torch's pip-installed CUDA
# wheels only bundle the runtime). This cluster has no CUDA toolkit's nvcc on PATH
# by default. The fallback sampler is not "non-JIT" -- it's still JIT-compiled, via
# Triton -- but Triton bundles its own self-contained compiler backend instead of
# shelling out to an external nvcc, so it doesn't hit this.
export VLLM_USE_FLASHINFER_SAMPLER=0

# vllm 0.28's core C extension (vllm/_C_stable_libtorch.abi3.so) and its cutlass-dsl
# [cu13] kernels are built against CUDA 13 and dlopen libcudart.so.13, which pip ships
# in the venv under nvidia/cu13/lib but leaves off every loader path (the extension's
# RPATH is $ORIGIN-only and NVIDIA's cuda wheels don't self-register). It then resolves
# only on nodes whose OS image happens to register a system CUDA-13 in ldconfig (login,
# some A100 nodes) and fails with "libcudart.so.13: cannot open shared object file" on
# nodes that register only CUDA-12 -- e.g. the H200 partitions m13h/eng (confirmed on
# m13h-2-2). Point the loader at the venv's own cu13 libdir so resolution no longer
# depends on the node image; the driver there already supports CUDA 13. Safe alongside
# torch's cu12 libs: cuda sonames are major-versioned (libcudart.so.12 vs .13), so a
# cu12 consumer never picks these up, and no soname here collides with the rest of venv.
export _SELFCONCEPT_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

cat > ./.env/etc/conda/deactivate.d/vllm_env_fixes.sh <<'EOF'
unset LD_PRELOAD
unset VLLM_USE_FLASHINFER_SAMPLER
if [ -n "${_SELFCONCEPT_OLD_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$_SELFCONCEPT_OLD_LD_LIBRARY_PATH"
else
    unset LD_LIBRARY_PATH
fi
unset _SELFCONCEPT_OLD_LD_LIBRARY_PATH
EOF
