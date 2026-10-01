# Measuring the self/other gap

This is the measurement tool for work-order step 1 in [PLAN2.md](PLAN2.md). It runs fixed prompts through the base model and optional interventions. It generates no answers and makes no behavioral claim.

The older `latent_soo.py` remains unchanged so its historical outputs keep their original meaning.

## Input pairs

Use a JSONL file with one pair per line. Each row needs these fields:

```json
{"id":"report-01","family":"report-explanation","split":"development","kind":"self_other","self_prompt":"You are writing a report. Explain your decision to yourself.","other_prompt":"You are writing a report. Explain your decision to a colleague."}
```

This is a format example, not a proposed evaluation set.

- `id`: a unique, stable pair ID.
- `family`: the shared situation or wording family. Closely related variants belong together. Uncertainty intervals resample whole families.
- `split`: `fit`, `development`, or `final`. A family cannot cross splits.
- `kind`: `self_other` or `nonsocial`. The latter uses the same two prompt fields for its two members; results stay separate.

The file may contain all three splits. `--split` selects the measured one. The runner rejects repeated IDs, families crossing splits, and identical prompts crossing splits. It cannot determine semantic overlap or whether an existing vector/adapter was actually fitted on the declared fit set. Check those against training records before calling a result held out. A development-only file records no claim of verified training separation.

The old probe files need explicit IDs, families, and second-study split assignments before use. Their `vocab_split=test` label does not make them fresh final-test data. Do not assign each wording variant a separate family to obtain narrower intervals.

## Run

From the repository root, with a prepared pair file:

```bash
SOO_CHAT_KWARGS='{"enable_thinking": false}' .venv/bin/python -m selfconcept.soo.measure_overlap \
  --model Qwen/Qwen3.8-27B \
  --probes path/to/declared_pairs.jsonl --split development \
  --layer 31 --device-map auto --dtype bfloat16 \
  --vectors path/to/existing_vectors.pt --token-mode last \
  --mode add --alpha 10 --positions all \
  --random-seeds 0 1 2 \
  --out experiments/soo/results/study2/overlap-development-add
```

Use the same vector, strength, chat settings, and position mask as the behavioral cell being compared. Change to `--mode project --alpha 1` for single-direction projection. Omit `--vectors` for base-only measurement, or add `--adapter path/to/local/adapter` to compare an existing adapter separately against base. Adapter and steering conditions are not combined. Run each adapter in its own output directory.

`--endpoint-offset N` reads the activation N valid tokens before the last prompt token at every capture site, for example the last user-content token instead of the assistant-header token. The saved token records show the actual token; check that it is matched within each pair before comparing. The intervention's position mask is unaffected.

`--layer` uses zero-based indices. By default, the runner measures the intervention block and every later block. `--residual-layers 31 32 63`, for example, can select fewer blocks if those indices exist. The intervention block is always included. `--revision` selects a model/tokenizer revision. CPU and float32 are the defaults; choose the experiment's actual precision and device explicitly.

Output directories must be new. Use a separate directory for each run; existing results cannot be overwritten or resumed silently.

## What is measured

The same rendered and tokenized inputs are reused for every condition, with dropout disabled and no truncation. Each capture reads the final non-padding token of each prompt independently. This can be an assistant-header token. The saved token records show exactly what was read.

Capture sites are:

1. The chosen attention output immediately before the intervention hook.
2. That output immediately after the hook.
3. The residual stream at the end of the chosen block and later blocks.
4. The decoder's final normalized representation.

For an adapter, both hook captures already include the adapter. Compare the separate `base` and `adapter` conditions to measure its effect. “Before hook” does not mean “before adapter.”

The primary gap is the mean squared difference between the two members, averaged over activation dimensions and pairs. The report also includes the gap between population means, activation norms, variation across prompts, and actual squared activation displacement from base. Distances use float32 arithmetic even when the model runs at lower precision. No norm-based denominator is applied.

Changes from base have paired, family-bootstrap 95% intervals. These describe uncertainty across the declared families; they do not address model seeds, dataset construction bias, or multiple comparisons. A single family receives no interval. Base and intervention use the same pairs. Nonsocial pairs are reported separately.

Additive steering should preserve the immediate paired difference, up to numerical rounding. Projection should remove its selected component at strength 1. Neither identity guarantees a downstream or behavioral result. In bfloat16, subtraction and projection have rounding error; use the inactive condition and saved tensors to judge its scale.

## Saved files

| File | Contents |
|---|---|
| `summary.json` | Metrics by condition, pair kind, and capture site |
| `pairs.jsonl` | Individual gaps, norms, displacement, and paired changes |
| `activations.pt` | Float32 CPU endpoint tensors, shaped `[pairs, 2, hidden]`; member order is self then other |
| `tokens.jsonl` | Rendered prompts, valid input IDs, and actual endpoint tokens |
| `manifest.json` | Pair order and declared splits, settings, versions, revisions, input/source hashes, and capture paths |

`manifest.json` is written last and marks a completed run. No generated completions exist for this forward-only measurement.

## Software checks

The exploratory subspace grid is described in [SUBSPACE_PILOT.md](SUBSPACE_PILOT.md). Add `--subspace-ranks 1 2 4 8 --subspace-strengths 0 0.5 1 --random-seeds 0 1 2` with a pair file containing both self/other and nonsocial fit rows. The runner fits on those rows only and measures development rows. It adds mean-direction projection, fitted and random subspaces, and an inactive subspace hook. `--checkpoint-conditions` saves each completed condition while the run is in progress; it does not enable automatic resume. Fitted directions, fit activations, and random bases are saved in `subspace_fit.pt`.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_soo*.py' -v
```

The tests use controlled tensors and tiny randomly initialized Llama and Qwen hybrid models on CPU. They check padding, hook order, final-representation capture, inactive steering, fixed additive shifts, projection, positional masking, adapter comparison, paired reporting, split validation, and output provenance. Qwen coverage includes both linear and full attention. An artificial gate checks the capture interface and the zero-gate identity; a trained probe gate is still future work.

Passing these checks establishes that the measurement code works on these examples. It is not evidence about overlap or honesty in the experimental models.
