# Dose and position round: how strong must the band's mean `v_proj` offset be, and where must it sit?

Written 2026-09-29, before launching. Exploratory. Follows the band-steering round recorded in [FINDINGS2.md](FINDINGS2.md): on gemma-4-12B (band 19–23), Mistral-7B (5–13) and gemma-4-31B (20–31), the band's mean `v_proj` delta over all prompt tokens, added as a fixed offset at every position with α = 1, reproduces the band's LoRA on Mistral and 31B and nearly on 12B (mirrored perspectives 87%); α = 2 damages every model; the same offset added only from the last prompt token onward does nothing; the matched-norm random offset is null on 31B, partial and mostly confused on Mistral, and strong on 12B.

## Questions

1. Dose. Between α = 1 and α = 2, where does the offset stop reproducing the band and start damaging the model, and does the matched random offset catch up with the real one as α grows?
2. Position. The offset works at every position and fails from the last prompt token onward. Is it enough on the user message's own tokens (the scenario and its answer-format sentence), is the whole prompt needed (adding the chat template's tokens), or must the generated tokens carry it too?

## Intervention

As in the band-steering round (`--band-deltas`, `selfconcept.soo.steering.steer_modules`): the band's `all` mean `v_proj` deltas, re-measured on the 156 training prompts in each job, added as α times a fixed vector to every band layer's value projection on the base model, at the masked positions. New position mode `content` (`--steer-positions content`): only the tokens of the user message's content, found as the tokens overlapping the message text in the chat-templated prompt (checked on all four tokenizers to match the chat template's own token ids, and to start at the message's first word); no chat-template tokens and no generated tokens. `prompt` is the existing mode: every prompt token before the chat template's generation header, and nothing after it.

Conditions, both orientations, all three scenarios, n = 250:

- `base`, `band-v`: same-day references (`band-v` is the band's own LoRA subset, the reference for the rule).
- `v-all-a1`: the band-steering round's reproducing cell, re-run as a same-day anchor.
- `v-all-a1.25`, `v-all-a1.5`, `v-all-a1.75`: the dose.
- `rand-all-a1.25`, `rand-all-a1.5`, `rand-all-a1.75`: seed-0 random directions of the same per-layer norm at each dose (the same draw as the band-steering round's `rand-all` cells).
- `v-prompt-a1`: the offset on every prompt token, none on the generation header or the generated tokens.
- `v-content-a1`, `v-content-a2`: the offset on the user message's tokens only, at α = 1 and 2 (fewer positions may need a larger dose).
- `rand-content-a1`: the seed-0 random offset on the same tokens.

No gap measurement this round: the gaps are read at the last prompt token, which the `content` and `prompt` offsets reach only through attention, and the band-steering round already measured the `all` settings.

## Fixed setup

As in the band-steering round: models and revisions from its launcher, bfloat16, one A100 per model, greedy decoding, the first study's suffix (`room_only` for gemma-4, `i_would` for Mistral), 100 new tokens, rule-based classifier with raw completions saved, intent split over every Mistral main cell, frozen snapshot with hashes, software checks (`tests/test_soo_dose_kimi.py` adds checks of the content mask during prefill and generation), `submission.json`. Launcher: `scripts/launch_steer_round.py --round dose`.

## Readout and decision, fixed in advance

The layer round's rule with `band-v` as the reference: **reproduces** when in both orientations the main deceptive rate is within 10 points of `band-v`, refusal-plus-other is at most 10 points above `band-v`'s, and perspectives correct is at least 90%; **damage** when refusal-plus-other exceeds 20 points or perspectives falls below 80%; **null** when the main deceptive rate is within 10 points of `base` in both orientations; otherwise partial. On Mistral a reproduction also has to be mostly clean honest responses by the intent split.

- Dose: the working window is the set of α at which `v-all` reproduces without damage. At each α, the direction holds only if the matched random cell is null (or, on Mistral, partial with mostly confused responses). If the random cell reproduces at some α, report that α as the size at which any offset suffices on that model.
- Position: `v-content-a1` (or `-a2`) reproducing with `rand-content-a1` null means the offset is needed only on the user message's tokens. If only `v-prompt-a1` reproduces, the template's tokens are needed too. If neither reproduces while `v-all-a1` does, the offset must also reach the generation header or the generated tokens. Report partial cells as partial.
- `v-all-a1` must reproduce again on Mistral and 31B; if it does not, the round's other cells are read against the same-day anchor and the discrepancy is reported first.

No honesty claim follows from any outcome.
