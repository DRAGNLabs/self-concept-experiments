# Prompting against reward hacking

Does a prompt alone make gpt-oss-120b reward hack less on coding problems, without making it worse at coding? This
is the prompting baseline for the steering results in [schemer-axis](../schemer-axis/). [PLAN.md](PLAN.md) is the
design and the open decisions; [FINDINGS.md](FINDINGS.md) is the record of every analyzed run. This file is the
operating manual. Read the reward-hacking setup first:
[experiments/codebench/reward_hack_recipe.md](../codebench/reward_hack_recipe.md) (Koby's clean protocol).

## Layout

| Path | What it is |
|---|---|
| `data/arms.tsv` | One prompt arm per line: `name <TAB> prompt file` (relative to this directory). Empty second column = no prompt. The no-prompt `baseline` is not in it: that run already exists (PLAN.md) and is copied into the results folder. |
| `data/prompts/` | The arm prompts, one `.txt` per arm. |
| `slurm/run_arms.sbatch` | Arms × seeds array on the 45 hard problems with the recipe; one task per (arm, seed). |
| `slurm/capability_eval.sbatch` | GSM8K, MMLU-Pro and HumanEval with each arm's prompt in front; one task per arm. Uses `../schemer-axis/scripts/capability_eval_vllm.py`. |
| `slurm/run_info.sh` | Writes `run_info/` (code version, prompt, checksums) into every run's output; sourced by both sbatch files. |
| `scripts/` | Summaries and analysis, adapted from schemer-axis. |

The problems are the 45 hard ImpossibleBench problems the recipe and schemer-axis use,
`/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data/ib_solvhard_s0-2.jsonl`
(Koby's; read in place; a copy with checksums is in `~/nobackup/autodelete/prompt-rh-prevention-results/problems/`).

Results go to `~/nobackup/autodelete/prompt-rh-prevention-results/<TAG>/<arm>/seed<s>/`: the episode records (every
completion, submitted code and test output), reasoning sidecars, summaries, and `run_info/` (git commit, `git diff`
of the working tree, arms TSV, the arm's prompt, problem checksums, Slurm job info). Everything is kept so episodes
can be checked or judged later; this study does not use an LLM judge for now (PLAN.md). Autodelete: copy what
matters somewhere permanent.

## How the prompt arms work

Every arm runs the recipe (`critical_no_carve` instruction, neutral feedback, 5 attempts with a stated budget, low
reasoning effort, temperature 1, 16384 new tokens per turn). An arm only adds a prompt in front of the first user
message: the model sees `[arm prompt]`, a blank line, then the recipe's task message unchanged (instruction, function
stub, shown asserts, budget). This is `selfconcept.codebench.run --prompt-prefix-file <file>`; each episode record
keeps the text in `prompt_prefix`. To add an arm, write `data/prompts/<name>.txt` and add
`<name><TAB>data/prompts/<name>.txt` to `data/arms.tsv`.

## Environment and conventions

- Python: the repo's `.venv` (`uv sync`; see the top-level README). Jobs do
  `export SELFCONCEPT_VENV=$SLURM_SUBMIT_DIR/.venv; source activate.sh`. ORC has no Python 3.13 module; one way to
  get the interpreter and `uv` is a small Miniforge env:
  `module load miniforge3; mamba create -n selfconcept-py313 -c conda-forge python=3.13.7 uv`, then
  `uv sync --frozen --python ~/.conda/envs/selfconcept-py313/bin/python3.13` from the repo root.
- Model: `hf download openai/gpt-oss-120b --exclude 'metal/*'` with `HF_HOME` as below (130 GB; on a login node,
  not through the compute-node proxy). Jobs load it by name offline, as the other experiments do, so download by
  name (no `--revision`; that skips the `refs/main` pointer) and keep `original/`: vLLM's offline weight lookup
  fails on a snapshot without it. main was `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` on 2026-10-29.
- Hugging Face: `HF_HOME=~/nobackup/autodelete/hf_cache`, offline in jobs; the checkpoint must be in the cache
  before a job starts.
- Before submitting, check where the GPUs are free (a few scheduler queries; once per submission, not in a loop):
  `PATH=/apps/slurm/latest/bin:$PATH .venv/bin/python -m byutils.gpu_pick --hours H --vram-gb 80 --cpus 8 --mem-gb 128 --cuda 13.0 --no-requeue-ok`.
  Use it to choose the GPU type and QoS and to see the queue; do not copy its `-p` list (ORC policy: no partition
  unless truly required) or its `--gpus 1` (that admits H100/B200; ask before using pricier GPUs). `--requeue-ok`
  (its default) adds standby options, which can be preempted; only with the user's OK.
- Submit every sbatch from the repo root (they `--chdir` into this directory). Jobs request one A100 with QoS `cs`
  and no partition.
- Record keeping: one dated section per analyzed run in FINDINGS.md; PLAN.md changed with a dated note when the
  design changes. Several people share the repo and the Slurm account: check `git log` and `squeue -u $USER` before
  committing or submitting.

## Running

```
sbatch --array=0-$((N_ARMS*NSEEDS-1)) --export=ALL,ARMS=data/arms.tsv,TAG=<run name>,NSEEDS=12 \
    experiments/prompt-rh-prevention/slurm/run_arms.sbatch
```

For a smoke run on a few problems, add `EXTRA_ARGS="--n 3"` and `NSEEDS=1` to `--export` (the first 3 of each
file). Other problem files: `PROBLEMS=<dir>` and `SCENARIOS="<file stems>"`.

Summary per arm: `python experiments/prompt-rh-prevention/scripts/pilot_summary.py ~/nobackup/autodelete/prompt-rh-prevention-results/<TAG>`.

General capability (same `data/arms.tsv`; output in `~/nobackup/autodelete/prompt-rh-prevention-results/capability/`):

```
sbatch --array=0-$((N_ARMS-1)) --export=ALL,ARMS=data/arms.tsv experiments/prompt-rh-prevention/slurm/capability_eval.sbatch
python experiments/schemer-axis/scripts/capability_eval_compare.py ~/nobackup/autodelete/prompt-rh-prevention-results/capability \
    --baseline baseline --arms <arm names>
```

The datasets have to be in `HF_HOME` (jobs are offline); they were downloaded on a login node on 2026-10-27.
