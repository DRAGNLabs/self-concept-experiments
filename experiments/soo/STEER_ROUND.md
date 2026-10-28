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

## Addendum, 2026-09-29, before launching: gemma-4-31B

Written after the 12B and Mistral jobs finished (not yet written up in FINDINGS2) and after the second bisection on gemma-4-31B ([BISECT2_ROUND.md](BISECT2_ROUND.md)) found its band: the seed-0 adapter's `v_proj` deltas at layers 20–31 reproduce the adapter on main in both orientations (0% / 2% deceptive, perspectives 100%, no refusals), 22–31 and 24–31 are partial, and every range ending below 31 is null. Layers 20–31 hold 10 `v_proj` modules; the other two are gemma-4 global-attention layers, which have none, and carry only their `q_proj` means.

Same design, conditions, gap measurement and rule as above, with `band-v` = 20–31 @ `v_proj` (top 31, training layer 32, last layer 59). No condition is added or dropped, so the three models stay comparable. Context the rule does not change: on gemma-4-12B the seed-0 random offsets at every position also moved the room task (53% / 84% deceptive at α = 1, 0% at α = 2), so on 31B the matched random cells decide whether any `v-all` reproduction is attributed to the direction, as the second bullet of the decision list already says. One random seed, as on the other two models.

## Addendum, 2026-09-29, before launching: Qwen3.8-27B

Written after the Qwen localization round ([QWEN_LOC_ROUND.md](QWEN_LOC_ROUND.md), FINDINGS2): on Qwen3.8-27B L32 the effect sits in layers 24–31 (all modules reproduce; 28–31 partial), in the fused Gated DeltaNet `in_proj_qkv` deltas (below-train @ `in_proj_qkv` reproduces; its value slice alone is partial at 42% / 56% deceptive; its query-key slice alone and the full-attention layers' `q_proj`/`v_proj` are null). There is no separate `v_proj` band on this model, so the steered module is the fused projection; the band's two full-attention layers (27, 31) are not steered, since their modules are null below L32 as a group.

Band deltas are measured on `in_proj_qkv`, `q_proj` and `v_proj` at layers 24–31, with each fused projection's query, key and value channel ranges recorded (read from the model). A slice offset (`--band-module in_proj_qkv.v` or `.qk`) is the mean delta with the other channels zeroed; its random control is confined to the same channels.

Conditions, both orientations, three scenarios, n = 250 (reference for the rule: `band-qkv`):

- `base`, `adapter-seed0`; `band-qkv` (the adapter's `in_proj_qkv` deltas at 24–31 only, the reference) and `band-qkv-v` (their value slice only); neither was run in the localization round.
- `qkv-resp-a1`: the `last` mean on the whole fused output, from the last prompt token onward.
- `qkv-all-a1`, `qkv-all-a1.5`, `qkv-all-a2`: the `all` mean on the whole fused output at every position.
- `v-all-a1`, `v-all-a2`: the value slice only; `qk-all-a1`: the query-key slice only (this model's counterpart of the `q_proj` control).
- `rand-all-a1`, `rand-all-a2`: seed-0 random directions of the whole fused offset's per-layer norm.
- `qkv-content-a1`, `qkv-prompt-a1`: the `all` mean on the user message's tokens only, or on every prompt token and no generated token ([DOSE_ROUND.md](DOSE_ROUND.md)).

Gaps as in the design above at the band's top layer (31, a full-attention layer), the training layer and the last layer, under the whole, value-slice and query-key-slice offsets with the seed-0 random control of the whole offset, the `band-qkv` subset, for the `resp` α = 1 and `all` α = 1 and 2 settings, and under the full adapter.

Decision list as above, with `band-qkv` as reference. Added for this model: `v-all` reproducing while `qk-all` is null would put the steering effect in the value channels even though the LoRA needed both; `qkv-all` reproducing while neither slice does would mean the fixed offset, like the LoRA, needs both parts. One random seed.
