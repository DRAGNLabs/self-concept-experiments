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
- ROOMS tasks, which test binding across entity types with one shared frame,
  "<entity> is in the <room>." → "Where is <entity>? … in the ___". The answer is the room itself.
  Each has a 200-context eval file and a disjoint 500-context `_estimate` file:

  | task | example context |
  |---|---|
  | `people` | "Alice is in the kitchen. Bob is in the garage." |
  | `animals` | "The dog is in the kitchen. The fox is in the garage." |
  | `objects` | "The key is in the kitchen. The lamp is in the garage." |
  | `i_you` | "I am in the kitchen. You are in the garage." (same speaker throughout) |
  | `animal_object` | "The dog is in the kitchen. The key is in the garage." (living + non-living) |

  ```bash
  python -m selfconcept.binding_ids.datagen --task animals
  python -m selfconcept.binding_ids.datagen --task animals --n 500 --seed 1 --name animals_estimate --exclude animals
  ```

  Pools are in `vocab.py`: single tokens in the OLMo-2 and Llama-3.2 tokenizers, with
  strong room associations (cow/barn, knife/kitchen) left out. For two-pool tasks the order
  alternates, so each pool comes first in exactly half the contexts.
  `scripts/compare_deltas.py` compares the binding vectors of different tasks.

### Running

All commands from the repo root.

**On the cluster:** download the model once on a login node (compute nodes are offline), then submit:

```bash
python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/llama3.2-1b.yaml --slurm.job_type local-download
python -m selfconcept.binding_ids.run --config experiments/binding_ids/configs/llama3.2-1b.yaml
```

Llama is gated on Hugging Face; log in with `huggingface-cli login` first.

**Request minimal compute.** Use the cheapest GPU that fits the model, one GPU, and time/memory
limits sized to the job. On the `cs` QoS that means `gpus_per_node: a100:1` (partition `cs`,
80GB A100s, enough for OLMo-2-13B, billing weight 224 per GPU). The H100s (`cs2`, 308) and
B200s (`cs3`, 640) cost more and aren't needed for these models. Check with
`scontrol show partition <name> | grep TRESBillingWeights`.

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
   `_e0`/`_e1` split by queried entity. Stricter variants:
   - `raw_accuracy_*`: uncalibrated; the model prefers the expected in-context capital
   - `vocab_accuracy_*`: the expected capital is the model's top-1 token over the whole vocabulary

   If binding IDs work as the paper describes, all should be high.

Results are written to `results/<run>/mean_interventions.json`, with per-query log probs and top
tokens in `results/<run>/mean_interventions_queries.jsonl` (to recompute any metric offline), and
printed in the Slurm log.

### Layout

- `configs/` — one yaml per model/run
- `data/` — generated binding-task datasets (committed)
- `scripts/` — one-off analysis and utility scripts
- `results/`, `slurm-logs/` — run outputs (gitignored)
