# self-concept-experiments

## Environment

uv-managed (see `pyproject.toml`).

- Create/sync the env: `uv sync` — creates `./.venv` from `pyproject.toml` + `uv.lock`.
- Add/change deps: edit `pyproject.toml`, then `uv sync` (commit the updated `uv.lock`).

To run anything that needs the cluster-specific runtime fixes (vLLM's nvcc/cu13
workarounds), source `activate.sh` instead of the bare venv activate — it activates
`./.venv` and exports those fixes:

```bash
source ./activate.sh
```

The slurm scripts already do this (`source ../../activate.sh`).
