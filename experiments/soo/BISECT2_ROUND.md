# Second bisection round (gemma-4-31B): which part of layers 16–31 carries the effect?

Written 2026-09-28, before launching. Exploratory. Follows the bisection round recorded in [FINDINGS2.md](FINDINGS2.md): on gemma-4-31B L32 the seed-0 adapter's `v_proj` deltas at layers 16–31 (13 modules) reproduce the main-scenario effect exactly and layers 0–15 do nothing; within 16–31, layers 24–31 alone are partial (45% deceptive orig, 94% mirrored) and 16–23 alone are null. The bisection round's second branch applies: bisect 16–31 further, starting from 24–31 and its unions with the upper part of 16–23.

## Question

What is the smallest contiguous band within 16–31 whose `v_proj` deltas reproduce the adapter's main-scenario effect, and are the top layers of the range needed?

## Intervention

As in the bisection round: load the seed-0 adapter unmerged and swap every LoRA module outside the chosen set back for its frozen base layer (`selfconcept.soo.lora_subset`, `--adapter-modules v_proj`). Nothing is merged.

Conditions, both orientations, all three scenarios, n = 250, `v_proj` only unless stated:

- `base`, `adapter-seed0`: same-day references.
- `hi-half` (16–31), `q4` (24–31): last round's reproducing band and its partial upper half, re-run as same-day anchors.
- Growing 24–31 downward: `u22` (22–31), `u20` (20–31), `u18` (18–31).
- Trimming 16–31 from the top: `t29` (16–29), `t27` (16–27), `t25` (16–25).
- Middle: `mid` (20–27).
- Split: `split` (16–19 and 24–31), whether the lower quarter's contribution comes from its bottom.

Collapse measurement as before (training-layer spread on the 156 training prompts) under each subset.

## Fixed setup

As in the bisection round: gemma-4-31B-it at the round 2c revision, bfloat16, one A100, greedy, `room_only` suffix, 100 new tokens, scenarios `main` / `treasure_hunt` / `perspectives`, both orientations, rule-based classifier with raw completions saved, frozen snapshot with hashes, software checks, `submission.json`; complete runs are skipped on restart.

## Readout and decision, fixed in advance

The layer round's rule, unchanged (reproduces / null / damage / partial against `adapter-seed0` and `base`, with the perspectives floor).

- The smallest union `uK` that reproduces sets the band's lower edge; the largest trim `tK` that reproduces sets its upper edge. The band is their intersection.
- If only `hi-half` reproduces, the effect needs all sixteen layers together and the localization on 31B stops at 16–31.
- `split` reproducing while `u18` does not would mean the lower contribution is not contiguous; report it and stop bisecting.
- Collapse next to each verdict as before.

No honesty claim follows from any outcome.
