# Prompting against reward hacking

Does a prompt alone make gpt-oss-120b reward hack less on coding problems, without making it worse at coding? This
is the prompting baseline for the steering results in [schemer-axis](../schemer-axis/). [PLAN.md](PLAN.md) is the
design and the open decisions; [FINDINGS.md](FINDINGS.md) is the record of every analyzed run. This file is the
operating manual. Read the reward-hacking setup first:
[experiments/codebench/reward_hack_recipe.md](../codebench/reward_hack_recipe.md) (Koby's clean protocol).

## Layout

| Path | What it is |
|---|---|
| `data/arms.tsv` | One prompt arm per line: `name <TAB> extra args for selfconcept.codebench.run`. Empty args = the recipe as is (`baseline`). |
| `slurm/run_arms.sbatch` | Arms × seeds array on the 45 hard problems with the recipe; one task per (arm, seed). |
| `scripts/` | Summaries and analysis, adapted from schemer-axis. |

Outputs go to scratch, `~/nobackup/autodelete/self-concept-experiments/prompt-rh-prevention/` (autodelete: copy what
matters). Shared data (Koby's): `/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/` (`MIRROR`), with
`rh_hillclimb/data/ib_solvhard_s0-2.jsonl` (the problems) and `rh_hillclimb/judge_rubric_v3.md` (the hack rubric).

## How the prompt arms work

Every arm runs the recipe (neutral feedback, 5 attempts with a stated budget, low reasoning effort, temperature 1,
16384 new tokens) and changes only the task instruction above the function stub:
`--impossible-prompt minimal|critical|critical_no_carve|strict` (`selfconcept.codebench.harness`). The recipe's own
instruction is `critical_no_carve`, so the `baseline` arm passes no extra args.

## Environment and conventions

- Python: the repo's `.venv` (`uv sync`; see the top-level README). Jobs do
  `export SELFCONCEPT_VENV=$SLURM_SUBMIT_DIR/.venv; source activate.sh`.
- Hugging Face: `HF_HOME=~/nobackup/autodelete/hf_cache`, offline in jobs; the checkpoint must be in the cache
  before a job starts.
- Submit every sbatch from the repo root (they `--chdir` into this directory). Jobs request one A100 with QoS `cs`
  and no partition.
- Record keeping: one dated section per analyzed run in FINDINGS.md; PLAN.md changed with a dated note when the
  design changes. Several people share the repo and the Slurm account: check `git log` and `squeue -u $USER` before
  committing or submitting.

## Running

```
sbatch --array=0-$((N_ARMS*NSEEDS-1)) --export=ALL,ARMS=data/arms.tsv,ROOT=<dir>,NSEEDS=12 \
    experiments/prompt-rh-prevention/slurm/run_arms.sbatch
```

For a smoke run on a few problems, add `EXTRA_ARGS="--n 3"` and `NSEEDS=1` to `--export`.
