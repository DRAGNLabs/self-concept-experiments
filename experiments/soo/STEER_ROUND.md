# Band-steering round: does the band's mean `v_proj` delta work as a fixed steering offset?

Written 2026-09-28, before launching. Exploratory. Follows the bisection round recorded in [FINDINGS2.md](FINDINGS2.md): on gemma-4-12B L24 the seed-0 adapter's `v_proj` deltas at layers 19–23 reproduce the room-task effect cleanly, and on Mistral-7B L16 the deltas at layers 5–13 reproduce it with clean honest responses. Each such delta is an input-dependent rank-r update, d_l(x) = s·B_l·A_l·x, on one layer's value projection. This round asks whether the band works as a set of fixed directions.

## Question

Does adding the band's *mean* `v_proj` delta, one fixed vector per band layer with no input dependence, reproduce the band's effect? If it does, is the direction what matters (random directions of the same per-layer norm do nothing) and is it specific to `v_proj` (the band's `q_proj` means do nothing)? And what does each intervention do to the self/other gap and the name-control gap at the band and downstream?

## Intervention

`scripts/extract_band_deltas.py` records, on the 156 training prompts (both members of the 78 pairs, chat-templated with the generation prompt), the delta each band LoRA module adds to its projection's output: its mean at the last prompt token (`last`, the position the last-token loss read) and its mean over all prompt tokens (`all`), with the root-mean-square spread of the per-token deltas around each mean relative to the mean's norm (how constant the delta is) and the base projection's output norm at the same positions (how large the offset is against what it is added to). Band steering (`selfconcept.soo.steering.steer_modules`, `--band-deltas` in `selfconcept.soo.evaluate`) adds α times the mean to the base model's projection output in every band layer at once, at the masked positions. No adapter is loaded in a steered cell.

Conditions, both orientations, all three scenarios, n = 250:

- `base`, `adapter-seed0`, `band-v`: same-day references; `band-v` is the bisection round's band (`v_proj` deltas at the band layers, everything else base) and is the reference the steered cells are judged against.
- `v-resp-a1`, `v-resp-a2`: the `last` means on `v_proj`, added from the last prompt token onward (`--steer-positions from_last`, the constant round's position mode, template-independent), α = 1 and 2.
- `v-all-a1`, `v-all-a2`: the `all` means on `v_proj`, added at every position (as the LoRA acts), α = 1 and 2.
- `rand-resp-a1`, `rand-resp-a2`, `rand-all-a1`, `rand-all-a2`: the same four cells with each layer's vector replaced by a random Gaussian direction of the same norm (seed 0; one draw per layer).
- `q-resp-a1`, `q-resp-a2`, `q-all-a1`: the band's `q_proj` means added to `q_proj`, same positions and token modes.

Gap measurement (`selfconcept.soo.measure_overlap`, the 32 self/other and 32 name-control development pairs of `data/subspace_pilot_pairs.jsonl`, last prompt token, bfloat16): the attention output of the band's top layer (the intervention site's read-out), the residual stream after the training layer, and after the last layer, plus the final norm, under `base`, `v_proj` band steering, its seed-0 random control, `q_proj` band steering, and the `band-v` LoRA subset, for each of the four (position, α) settings above, and once under the full adapter.

Bands: gemma-4-12B layers 19–23 (top 23, training layer 24); Mistral-7B layers 5–13 (top 13, training layer 16). Both from the seed-0 adapters used in the bisection round.

## Fixed setup

As in the bisection round: models and revisions from the round 2c launcher, bfloat16, one A100 per model, greedy decoding, the first study's suffix (`room_only` for gemma-4, `i_would` for Mistral), 100 new tokens, scenarios `main` / `treasure_hunt` / `perspectives`, both orientations, the rule-based classifier as primary readout with raw completions saved. Frozen snapshot with hashes, software checks (`tests/test_soo_band_steer.py` checks the delta measurement against the LoRA weights and the multi-layer hook), `submission.json`; complete runs are skipped on restart. On Mistral the intent-language split runs over every main cell at the end of the job.

## Readout and decision, fixed in advance

The layer round's rule with `band-v` as the reference. A condition **reproduces** when in both orientations its main deceptive rate is within 10 points of `band-v`, its refusal-plus-other rate is at most 10 points above `band-v`'s, and its perspectives correct rate is at least 90%. **Damage**: refusal-plus-other above 20 points or perspectives below 80%. **Null**: main deceptive rate within 10 points of `base` in both orientations. Otherwise partial. On Mistral a reproduction also has to be mostly clean honest responses by the intent split (the band's own were 75% / 91%).

- A `v` cell reproduces and its matched random cell (same positions, same α) is null: the band's mean delta is a working steering direction; the effect does not need the input-dependent part of the update. The smallest α that reproduces is the dose to carry forward.
- A `v` cell and its random cell both reproduce: a perturbation of that size at those layers is enough; the direction is not established.
- No `v` cell reproduces while `band-v` does: the fixed mean does not carry the effect; the input-dependent part of the deltas is needed (or the mean over training prompts is the wrong constant for the room prompts). The spread statistics say which; a spread near or above 1 means the deltas were never close to constant.
- `q` cells reproducing would mean the construction is not specific to the value path; report it as such.
- Gap: report each condition's self/other and name-control gap changes against base with their family-bootstrap intervals at the three sites. A direction that reduces the self/other gap more than the name-control gap at the band's read-out is the first such result in this study; a reduction of both by similar amounts is the collapse pattern of rounds 2–2c; no reduction under a reproducing cell dissociates the behavior from the gap. Descriptive only; no threshold is set.

No honesty claim follows from any outcome. A deceptive-rate drop is a behavioral change; on Mistral it has to pass the response reading before it counts as anything other than a confused answer.
