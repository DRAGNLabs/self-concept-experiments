# Bisection round: which layers below the training layer carry the adapters' effect on `v_proj`?

Written 2026-09-24, before launching. Exploratory. Follows the layer-localization round recorded in [FINDINGS2.md](FINDINGS2.md): on gemma-4-31B L32, gemma-4-12B L24, and Mistral-7B L16, the seed-0 adapter's LoRA modules below the training layer reproduce the room-task effect, `v_proj` alone reproduces it, the training layer's own modules and everything above it are inert, and no single layer's delta reproduces it. The effect is distributed over the `v_proj` deltas below the training layer. This round bisects that set.

## Question

Within layers 0 .. L−1, which contiguous band of `v_proj` deltas is sufficient for the adapter's main-scenario effect, and is any band necessary? Does the sweep's partial region (Mistral layers 5–13, 12B layers 19–23) carry it when its layers act together?

## Intervention

As in the layer round: load the seed-0 adapter unmerged and swap every LoRA module outside a chosen (layer, module) set back for its frozen base layer (`selfconcept.soo.lora_subset`, `--adapter-layers range:A-B` or `layers:...`, `--adapter-modules v_proj`). Nothing is merged.

Conditions, both orientations, all three scenarios, n = 250:

- `base`, `adapter-seed0`: same-day references.
- `below-v`: `v_proj` deltas in layers 0 .. L−1. This is the root of the bisection; the layer round tested `below-train` (both modules) and `v-only` (all layers) but not their intersection.
- Halves: `lo-half` = layers 0 .. L/2−1, `hi-half` = L/2 .. L−1, `v_proj` only.
- Quarters: `q1` .. `q4`, each L/4 layers, `v_proj` only.
- Sweep band and its complement, `v_proj` only: Mistral `band` = 5–13, `band-rest` = 0–4 and 14–15; gemma-4-12B `band` = 19–23, `band-rest` = 0–18. gemma-4-31B has no partial single layer, so no band condition.

Collapse measurement (`make_constants.py`, 156 training prompts): root-mean-square spread of the training layer's last-token attention output relative to its mean norm, for the base model and each adapter subset.

## Fixed setup

As in the layer round: models and revisions from the round 2c launcher, bfloat16, one A100 per model, greedy decoding, the first study's suffix (`room_only` for Gemma-4, `i_would` for Mistral), 100 new tokens, scenarios `main` / `treasure_hunt` / `perspectives`, both orientations, the rule-based classifier as primary readout with raw completions saved. Frozen snapshot with hashes, software checks, `submission.json`; complete runs are skipped on restart. On Mistral the intent-language split runs over every main cell at the end of the job.

## Readout and decision, fixed in advance

The layer round's rule, unchanged. A condition **reproduces the adapter** when in both orientations its main deceptive rate is within 10 points of the same-day `adapter-seed0`, its refusal-plus-other rate is at most 10 points above the reference's, and its perspectives correct rate is at least 90%. **Damage**: refusal-plus-other above 20 points or perspectives below 80%. **Null**: main deceptive rate within 10 points of `base` in both orientations. Otherwise partial. On Mistral a reproduction also has to be mostly clean honest responses by the intent split, as the full adapter's are (85–97%).

- `below-v` must reproduce; if it does not, `q_proj` contributes below the training layer and the bisection restarts on both modules.
- One half reproduces and the other is null: the next round bisects that half further, and its quarters here already say which quarter to start from.
- Both halves partial and no quarter reproduces: the effect needs layers from both halves acting together; the next round tests unions of the strongest quarters rather than smaller bands.
- `band` reproduces and `band-rest` is null: the sweep's partial region is the locus; the next round studies the band's `v_proj` deltas directly (their rank-r update as a set of steering directions).
- Collapse: report the training-layer spread next to each verdict as before; a reproducing band with spread near base, or a collapsing band with base behavior, dissociates the two.

No honesty claim follows from any outcome. A deceptive-rate drop is a behavioral change; on Mistral it has to pass the response reading before it counts as anything other than a confused answer.
