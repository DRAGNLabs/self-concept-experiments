# Recreating "How do Language Models Bind Entities in Context?"

Recreation of Feng & Steinhardt 2023 ([arXiv:2310.17191](https://arxiv.org/abs/2310.17191)).

## 2. Model choices

| Role in paper | Our substitute | Why |
|---|---|---|
| TODO | TODO | TODO |

## 3. Repo layout

```
src/selfconcept/binding_ids/
  tasks.py           # binding task templates
  vocab.py           # entity / attribute pools
  datagen.py         # instantiate templates -> JSONL datasets
  activations.py     # capture and patch activations at entity/attribute positions
  interventions.py   # factorizability, position-independence, mean interventions
  binding_vectors.py # estimate binding vectors and study their geometry
  evaluate.py        # task accuracy with and without interventions
  analysis.py        # aggregate runs -> tables and plots
experiments/binding_ids/
  configs/           # yaml per model/run
  data/              # generated datasets (committed)
  scripts/           # one-off analysis/utility scripts
  slurm/             # cluster job scripts
  results/           # run outputs (gitignored)
```

## 4. Phases

**Phase 1 — data.** TODO

**Phase 2 — baseline accuracy.** TODO

**Phase 3 — factorizability and position independence.** TODO

**Phase 4 — mean interventions / binding vectors.** TODO

**Phase 5 — analysis.** TODO

**Out of scope:** TODO

## 5. Adaptation decisions (paper → us)

- TODO

## 6. Connection to self-concept

TODO: how binding IDs relate to the self/other questions in this repo (e.g. the SOO experiment).

## 7. Success criteria

- TODO
