# self-concept-experiments

## Environment

Mamba-managed (see `environment.yaml`), not uv.

- First time: `./setup_env.sh` — creates `./.env` and applies cluster-specific fixes.
- After adding/changing deps in `environment.yaml`: `./update_env.sh` — updates `./.env` in place (faster than a full rebuild).
- `./fix_env.sh` — the fix logic shared by both above; only run directly if the env's activation hooks need reapplying for some other reason.

Then `mamba activate ./.env`.
