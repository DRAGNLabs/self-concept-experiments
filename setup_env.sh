#!/bin/bash
# One-time environment setup (rerun after any `rm -rf .env`). Run from the repo root:
#   ./setup_env.sh
# Already have ./.env and just added new deps to environment.yaml? Use
# ./update_env.sh instead -- it's much faster (updates in place, no full rebuild).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mamba env create -f environment.yaml -p ./.env

./fix_env.sh

echo "Environment ready at ./.env (mamba activate ./.env)"
