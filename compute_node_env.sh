#!/bin/bash
# Environment for jobs on compute nodes, which have no internet access: the shared HF cache, read offline, and the
# Harmony vocab copy in scratch (load_harmony_encoding otherwise downloads it into node-local /tmp). Source it from an
# sbatch script, next to activate.sh:
#   source ../../compute_node_env.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
