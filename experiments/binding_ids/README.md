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

1. Estimate binding vectors: the mean activation difference between the second and first
   pair, at every layer, at the last token of each entity and each attribute (first 100 contexts).
2. On the remaining contexts, query each entity under four conditions: no intervention, swap the
   entities' binding IDs, swap the attributes', swap both. Context activations are frozen at their
   clean values plus the swap offsets.
3. `accuracy_<condition>` = how often the answer matches the binding-ID prediction: the original
   answer for `none` and `swap_both`, the other pair's answer for single swaps. If binding IDs
   work as the paper describes, all four should be high.

Results are written to `results/<run>/mean_interventions.json` and printed in the Slurm log.

### Layout

- `configs/` — one yaml per model/run
- `data/` — generated binding-task datasets (committed)
- `scripts/` — one-off analysis and utility scripts
- `results/`, `slurm-logs/` — run outputs (gitignored)
