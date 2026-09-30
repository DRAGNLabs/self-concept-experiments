# CoT-ness or assistant-axis outcome correlations

The runner has two mutually exclusive measurement modes. `--measurement cotness`
(the default) fits and validates the existing role probe. `--measurement
assistant-axis` loads a frozen assistant axis and measures signed residual-stream
projections; it never fits or loads a CoT-ness probe. Each run and output directory
contains only one measurement. The analysis reads the mode from `manifest.json`;
older runs without a mode are treated as CoT-ness.

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
HF_HUB_OFFLINE=1 .venv/bin/python -m selfconcept.correlation.run \
  --model gemma4-12b --out experiments/cotness/results/diagnostic \
  --n 2 --code-n 1 --scenarios main impossible_conflicting evilgenie

# Repeat analysis after grades arrive:
.venv/bin/python -m selfconcept.correlation.analyze <snapshot>/output/*/
```

Each model directory holds `manifest.json`, `probes/validation.json`, frozen
linear weights, `generations.jsonl`, compressed token traces, `outcomes.jsonl`,
files accepted by the existing benchmark judges, and `analysis.json` / `analysis.md`.
Only `manifest.json` stage `complete` indicates a finished model run. A validation
failure is recorded explicitly. No hypothesis conclusion follows merely from
successful job submission or a successful probe fit.

## Assistant-axis / ImpossibleBench mode

On an allocated GPU with the model cached locally, use a Python 3.13+ environment
(the commands below use this checkout's `.venv-313`):

```bash
HF_HUB_OFFLINE=1 .venv-313/bin/python -m selfconcept.correlation.run \
  --measurement assistant-axis \
  --model allenai/Olmo-3.1-32B-Think --family olmo \
  --assistant-axis /path/to/olmo32b-thinking/axis_response_only.pt \
  --axis-layers 32 \
  --out experiments/cotness/results/axis-impossible-olmo \
  --code-n 12 --code-max-new-tokens 16384 --max-attempts 3

# Optional code judgments use the existing judge and output format.
.venv-313/bin/python -m selfconcept.codebench.judge \
  --responses experiments/cotness/results/axis-impossible-olmo/base_impossible_*.jsonl \
  --batch-size 1 --skip-existing

.venv-313/bin/python -m selfconcept.correlation.analyze \
  experiments/cotness/results/axis-impossible-olmo
```

Use an axis produced for the **same model/checkpoint** by
`experiments/assistant-axis/scripts/pipeline/5_axis.py`: `axis.pt`,
`axis_all_tokens.pt`, or `axis_response_only.pt`. Both bare `[layers, hidden]`
tensors and dictionaries containing `axis` are supported. Dimensions, selected
layers, and finite nonzero directions are checked. A dictionary's optional
`model` field is checked against the model ID; legacy bare tensors have no model
identity metadata, so selecting a matching checkpoint is the caller's responsibility.
The axis is copied into the output directory and its SHA-256 recorded. Changing
the mode, axis contents, layers, or other configuration on resume is rejected.

Registered model keys such as `gemma4-12b` retain their pinned revisions and
families. Axis mode also accepts an HF model ID with `--family` (`olmo`, `qwen`,
`gemma`, or `muse`) and optional `--revision`; pin the revision to the checkpoint
used to compute the axis. CoT-ness retains its registered-model restriction.

Axis mode defaults to `impossible_original`, `impossible_conflicting`, and
`impossible_oneoff`; `--scenarios` can override this. The original split remains
a capability control and is excluded from reward-hacking correlations. Existing
harness outcomes and optional code-judge labels provide the same reward-hacking
proxies used by the CoT-ness pipeline. Truncated or missing outcomes are unscored.

Each token's score is `hidden @ (axis / ||axis||)`, using the post-decoder-layer
residual, with no centering, probability transform, or sign flip. `--axis-layers`
uses zero-based decoder indices; its first layer is primary. If omitted, the
runner uses the same fixed midpoint as CoT-ness (`num_layers // 2 - 1`). Select
layers before examining outcomes. Both modes summarize first-turn prompt, CoT,
and final-answer content separately. Prompt scores are the primary predictor;
generated regions are descriptive associations with the eventual task outcome.
The final sampled token is excluded because generation never processes it.

Axis runs write `assistant_axis.pt`, `manifest.json`, `generations.jsonl`, token
traces with `assistant_axis_layer_<layer>` arrays, `outcomes.jsonl`, and judge
inputs. They have no `probes/` directory. Analysis produces the same Spearman,
direction-preserving AUROC, bootstrap intervals, coverage, and length adjustment
as CoT-ness, with `measurement: assistant-axis` in the report. The pilot/follow-up
launch scripts above still implement the original CoT-ness experiment protocol;
use the direct runner for assistant-axis experiments.
