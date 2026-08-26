#!/bin/bash
# Update the existing ./.env after adding/changing deps in environment.yaml (faster
# than a full ./setup_env.sh rebuild). Run from the repo root:
#   ./update_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ ! -d ./.env ]; then
    echo "./.env doesn't exist yet -- run ./setup_env.sh instead." >&2
    exit 1
fi

mamba env update -f environment.yaml -p ./.env

# `mamba env update` re-runs the whole pip section, which re-resolves vllm's
# opencv-python-headless>=4.13 dependency and would silently reintroduce it.
./fix_env.sh

echo "Environment updated at ./.env"
