# CoT-ness and deception

See [FOLLOWUP.md](FOLLOWUP.md) for the current repair/expansion protocol and
[PLAN.md](PLAN.md) for the original exploratory pilot. Kimi is excluded: the
original manually supplied reasoning markers were unsupported.

From the repository root:

```bash
# Current follow-up: repair pilots, gated expansion, and per-model judges.
HF_HUB_OFFLINE=1 .venv/bin/python experiments/cotness/scripts/launch_followup.py
# Add --no-submit to freeze and validate without launching.

# Uses cached WikiText; reproduces the committed, article-disjoint corpus.
.venv/bin/python experiments/cotness/scripts/prepare.py

# Freeze code/data, validate cached model assets, and submit the baseline pilot.
.venv/bin/python experiments/cotness/scripts/launch.py
# Prepare without submission, or restrict models:
.venv/bin/python experiments/cotness/scripts/launch.py --no-submit --models gemma4-12b

# Direct execution on an allocated GPU, for an isolated diagnostic:
HF_HUB_OFFLINE=1 .venv/bin/python -m selfconcept.cotness.run \
  --model gemma4-12b --out experiments/cotness/results/diagnostic \
  --n 2 --code-n 1 --scenarios main impossible_conflicting evilgenie

# Repeat analysis after grades arrive:
.venv/bin/python -m selfconcept.cotness.analyze <snapshot>/output/*/
```

Each model directory holds `manifest.json`, `probes/validation.json`, frozen
linear weights, `generations.jsonl`, compressed token traces, `outcomes.jsonl`,
files accepted by the existing benchmark judges, and `analysis.json` / `analysis.md`.
Only `manifest.json` stage `complete` indicates a finished model run. A validation
failure is recorded explicitly. No hypothesis conclusion follows merely from
successful job submission or a successful probe fit.
