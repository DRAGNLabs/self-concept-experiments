# self-concept-experiments
For exploring the possibility of a self-concept in LLMs with various experiments.

## Self-Other Overlap (SOO) recreation

Recreation of the LLM experiments from [Carauleanu et al. 2024](https://arxiv.org/abs/2412.16325)
("Towards Safe and Honest AI Agents with Neural Self-Other Overlap") using the OLMo 2 model
family. See [PLAN.md](PLAN.md) for the full design.

### Setup

```bash
conda env create -f environment.yaml
conda activate soo
```

### Data

Committed under `data/`, regenerable with:

```bash
python -m selfconcept.soo.datagen
```

- `data/train_soo_pairs.jsonl` — 78 self/other prompt pairs (3 templates x 26 train items)
  for SOO fine-tuning
- `data/eval/<scenario>.jsonl` — 250 held-out instantiations each of the main burglar
  scenario, 7 generalization variants, 2 extended scenarios, and the Perspectives control
- `data/latent_probes.jsonl` — 52 self/other pairs for latent-overlap (layer-wise MSE)
  measurement
