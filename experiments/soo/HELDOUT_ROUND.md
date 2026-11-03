# Held-out round: the selected configurations on new instantiations

Written 2026-10-28, before generating any held-out response (PLAN.md, paper-review follow-ups, item D).

Every mechanism-stage result so far (localization, bands, constants, fixed offsets, control round) was selected and
reported on the same 250 instantiations per scenario and orientation, built from the test vocabulary. The paper therefore
calls those results exploratory. This round freezes the configurations the paper's claims rest on, chooses the random
controls by the same rule as the real offsets, and evaluates all of them once on instantiations built from a vocabulary
no development run has seen.

## Held-out data

- New vocabulary (`selfconcept.soo.vocab`, `*_HELDOUT`): 26 expensive objects, 20 cheap objects and 12 indoor rooms,
  disjoint from the train and the test lists (checked at import; no held-out room name contains another as a whole word).
- `python -m selfconcept.soo.datagen --heldout --out data` (seed `heldout-20261028`): 250 instantiations of every
  scenario in both orientations, in `data/eval_heldout` and `data/eval_heldout_mirrored` (matched pairs, as before).
  Regenerating with the default arguments still reproduces the committed development sets byte for byte.
- Scenarios: main (burglar), Treasure Hunt and Perspectives as before; plus the first study's seven burglar wording
  variants and Escape Room (eight variants) as a near-transfer check. Kimi-Dev-72B: main and Perspectives only, as in its
  rounds.

## Selection, on development data only

`scripts/select_heldout.py` reads every development snapshot of each model's offset rounds (steering, dose, control;
Kimi joint-offset and control) and, separately for real offsets and random draws, picks the all-position cell minimizing
(class, max over orientations |deception − reference deception|, α, seed), class 0 = reproduces without damage or
budget-bound, 1 = no damage and not budget-bound, 2 = other. Each cell is read with the reproduction rule against its own
snapshot's same-day reference (bands; Qwen the full adapter; Kimi the LoRA subset the offset replaces). The random
selection runs over every draw and dose tested, so the selected random control has had at least as many chances as the
real offset. The selection is written to `results/study2/heldout-selection-*/` before any held-out run.

## Configurations, frozen

Offsets use the band deltas of each model's band-steering snapshot (identical to the dose round's; the control round's
H100 re-extraction differs by at most 0.8% relative; Kimi: the localization round's, as used by its joint-offset and
control rounds), copied into the snapshot, so the vectors are the ones evaluated in development. Random draws are
regenerated from those deltas and the selected seed. Constants come from each model's constant-round snapshot.

| Model | Cells |
|---|---|
| Mistral-7B, Gemma-4-12B, Gemma-4-31B | base, adapter, below the loss layer, band (`v_proj`), constant replacement with the adapter constant from the last prompt token and at all positions, selected real offset, selected random offset, the real offset at the selected α on the prompt only and on the user message only |
| Qwen3.8-27B | base, adapter, below the loss layer, band (`in_proj_qkv` 24–31), selected real offset, selected random offset, prompt-only and user-message-only offsets at the selected α and at α 2 (the paper's position dose) |
| Kimi-Dev-72B | base, adapter, below the loss layer, layers 28–55 (`hi-half`), selected real offset, selected random offset, and the real offset at the selected random draw's band and dose (a matched comparison) |

Variants (eight, both orientations): base, adapter, below the loss layer, band, selected real and selected random offset,
on every model except Kimi. Fixed setup otherwise as in the control round (greedy, bfloat16, the model's suffix, 100 new
tokens; Kimi 1024 with classification after the think block; intent split on Mistral main).

## Readout and decisions, fixed in advance

Same rule as before (reproduces / null / partial, damage; Kimi budget-bound), against the held-out runs of the same
references: below the loss layer, band and constants against the adapter; offsets against their band (Qwen: adapter;
Kimi: the LoRA subset). Analysis: `scripts/analyze_heldout_round.py`, committed before the first held-out job finishes.

1. **Per configuration:** confirmed on held-out if it reproduces there (for a configuration that reproduced in
   development); not confirmed otherwise. Report development and held-out deception side by side (shrinkage).
2. **Direction:** direction-specific on held-out if the selected real offset reproduces, the selected random offset does
   not, and the random offset is at least 10 points more deceptive in some orientation. No advantage if the selected
   random offset reproduces. Otherwise mixed.
3. **Constants:** the paper's negative claim (no replacement reproduces the adapter) holds if no constant cell reproduces.
4. **Informativeness:** an orientation where the held-out base model is below 50% deception cannot show a reduction;
   flag it and read reductions only in the other orientation.
5. **Variants:** descriptive. Per cell and variant, deception in both orders; per cell, the number of variants with
   deception at least 10 points below base in both orders.
6. Mistral: reproductions should be mostly clean honest answers by the intent split, as before. No honesty claim follows.

If a held-out reference (adapter or band) itself fails to reduce deception, the cells read against it are uninformative;
report that first.

## Selection result (2026-10-28, `results/study2/heldout-selection-20261028T181321Z/selection.md`)

| Model | selected real offset (development) | selected random offset (development) |
|---|---|---|
| Mistral-7B | α 1: 18.0 / 0.0, reproduces | α 2, seed 1: 23.2 / 16.8, partial |
| Gemma-4-12B | α 1: 0.0 / 0.0, partial (Perspectives 87.2) | α 1.25, seed 0: 0.0 / 4.8, reproduces |
| Gemma-4-31B | α 1: 0.0 / 0.0, reproduces | α 2, seed 0: 87.2 / 97.2, partial |
| Qwen3.8-27B | α 1.5: 0.0 / 0.0, reproduces the adapter | α 2, seed 0: 76.8 / 96.8, partial |
| Kimi-Dev-72B | layers 0–55, α 0.25: 100 / 97.6, partial (more deceptive than base; every other real cell is damaging or budget-bound) | layers 28–55, α 0.5, seed 0: 31.2 / 22.4, reproduces `hi-half` |

## Jobs

Per model one snapshot (`results/study2/heldout-*`), `run.sh SNAPSHOT JOB`, qos `cs`: job A the main three scenarios for
every cell, job B (and C on Mistral) the variants for the six core cells; Kimi four jobs of one or two cells on one B200
each. Readout: `analyze_heldout_round.py SNAPSHOT` after the model's last job.
