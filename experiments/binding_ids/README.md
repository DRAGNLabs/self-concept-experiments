# Binding IDs recreation

Recreation of the binding-ID experiments from [Feng & Steinhardt 2023](https://arxiv.org/abs/2310.17191)
("How do Language Models Bind Entities in Context?"). See [PLAN.md](PLAN.md) for the full design
and [FINDINGS.md](FINDINGS.md) for results.

Built on the [MIRROR pipeline](https://github.com/DRAGNLabs/MIRROR-Pipeline): models come from
MIRROR, the experiment is a `MirrorMetric`, and jobs are submitted through MIRROR's sbatch template.

### Setup

Separate from the SOO env, because MIRROR needs Python 3.12. From the repo root:

```bash
mamba env create -f experiments/binding_ids/environment.yaml
mamba activate binding_ids
pip install -e .
```

### Data

```bash
python -m selfconcept.binding_ids.datagen
```

- `data/capitals.jsonl` — 200 contexts, each binding two names to two countries
  ("Alice lives in the capital city of France. Bob lives in the capital city of Thailand."),
  queried for the capital of one name's country
- `data/capitals_estimate.jsonl` — 500 further contexts (none shared with `capitals.jsonl`) used
  only to estimate binding vectors:
  `python -m selfconcept.binding_ids.datagen --n 500 --seed 1 --name capitals_estimate --exclude capitals`

### Running

All commands from the repo root.

**On the cluster:** download the model once on a login node (compute nodes are offline), then submit:

```bash
python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/llama3.2-1b.yaml --slurm.job_type local-download
python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/llama3.2-1b.yaml
```

Llama is gated on Hugging Face; log in with `huggingface-cli login` first.

**Locally:** GPT-2 smoke test on CPU:

```bash
MIRROR_DATA_PATH=~/mirror_data python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/gpt2-local.yaml
```

### The experiment (mean interventions)

1. Estimate binding vectors on `capitals_estimate.jsonl`: the mean activation difference between
   the second and first pair, at every layer, at each entity's last token plus the token after it
   (" lives", as in the paper) and at each attribute's last token.
2. On 100 contexts of `capitals.jsonl` (rows 100–199), query each entity under seven conditions:
   no intervention; swap the entities' binding IDs, the attributes', or both; and the same three
   with each binding vector replaced by a random vector of equal norm (control). Context
   activations are frozen at their clean values plus the offsets.
3. `accuracy_<condition>` = how often the answer matches the binding-ID prediction: the original
   answer for `none`, `swap_both` and the random controls, the other pair's answer for single
   swaps. Answers are scored on each capital's first-token log prob, median-calibrated per answer
   slot (removes the model's bias toward e.g. the later-mentioned attribute, as in the paper);
   `raw_accuracy_*` skips calibration, and `_e0`/`_e1` split by queried entity. If binding IDs
   work as the paper describes, all should be high.

Results are written to `results/<run>/mean_interventions.json` and printed in the Slurm log.

### Layout

- `configs/` — one yaml per model/run
- `data/` — generated binding-task datasets (committed)
- `scripts/` — one-off analysis and utility scripts
- `results/`, `slurm-logs/` — run outputs (gitignored)
