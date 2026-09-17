# self-concept-experiments

1. Create the env:
The code below assumes there will be only 1 env for this folder.
```bash
mamba env create -f environment.yaml -p ./.env
mamba activate ./.env
pip install -e .
```

2. Activate the env:

```bash
mamba activate ./.env
```

3. As needed, update the env:

```bash
mamba env update -f environment.yaml -p ./.env --prune
```
