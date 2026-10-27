# Control round: more random draws, and position with the vector held fixed

Written 2026-10-27, before launching, in response to a review of the paper draft (Section 7, fixed offsets). Exploratory.

The band-steering, dose and Kimi joint-offset rounds ([FINDINGS2.md](FINDINGS2.md)) compared each extracted mean offset
with **one** matched-norm random draw (`--band-random-seed 0`) per model. Two of the paper's readings rest on that single draw:

- Gemma-4-31B and Qwen3.8-27B: the random draw is null where the real offset reproduces, read as "the direction matters".
- Gemma-4-12B (random α = 1.25 meets the rule) and Kimi-Dev-72B (random α = 0.75 over 0-55 gives 2.8 / 2.0% deception,
  random α = 0.5 over 28-55 meets the rule against `hi-half`): read as "no advantage for the extracted direction".

A single draw shows that some random direction does or does not work; it says nothing about random directions in general.
Separately, the position conclusion ("the offset must reach the prompt") compared prompt-only and content-only offsets,
which use the all-token mean, with `from_last` offsets that use the final-token mean, so vector and position changed together;
and on Qwen the prompt and content offsets were tested only at α = 1, where the all-position offset is itself partial.

## Questions

1. Random draws. At the doses the claims rest on, how many of five independent random draws (seed 0 from the earlier rounds,
   seeds 1-4 here) reproduce the reference, and how does their deception compare with the real offset's?
2. Position, same vector. With the all-token mean, does the offset still work when applied only from the last prompt token
   onward (`from_last`)? On Qwen, do the prompt-only and content-only offsets work at α = 2?

## Intervention

As in the band-steering and dose rounds (`--band-deltas`, `selfconcept.soo.steering.steer_modules`); band deltas re-measured
on the 156 training prompts in each job. Random cells use `random_matched_offsets` with seeds 1-4 (per-layer Gaussian direction
with the real offset's norm at that layer; on Qwen confined to the fused module's channels, as before).

Conditions, both orientations, main / treasure_hunt / perspectives, n = 250 (Kimi: main + perspectives, as in its rounds):

| Model | Same-day anchors | Random draws (seeds 1-4) | Position, all-token mean |
|---|---|---|---|
| Mistral-7B (v 5-13) | base, band-v, v-all-a1 | α 1, 2 | v-last-a1, v-last-a2 |
| Gemma-4-12B (v 19-23) | base, band-v, v-all-a1 | α 1, 1.25 | v-last-a1, v-last-a2 |
| Gemma-4-31B (v 20-31) | base, band-v, v-all-a1 | α 1, 1.5, 2 | v-last-a1, v-last-a2 |
| Qwen3.8-27B (in_proj_qkv 24-31) | base, band-qkv, adapter-seed0, qkv-all-a2 | α 1.5, 2 | qkv-last-a1, qkv-last-a2, qkv-prompt-a2, qkv-content-a2 |
| Kimi-Dev-72B (q+v) | carried over from the joint-offset round | rand-all α 0.75 (0-55), rand-hi α 0.5 (28-55) | — |

`v-last` = `--band-token-mode all --steer-positions from_last` (the earlier `v-resp` cells used `--band-token-mode last`).

## Fixed setup

As in the dose round: bfloat16, one GPU per model (Kimi: one B200, `--device-map auto`, 1024 new tokens, classification
after the think block), greedy decoding, the first study's suffix (`room_only`; Mistral `i_would`), 100 new tokens,
rule-based classifier with raw completions saved, intent split on every Mistral main cell, frozen snapshot with hashes,
software checks, `submission.json`. Launchers: `scripts/launch_steer_round.py --round control`, `scripts/launch_kimi_seeds.py`.

## Readout and decision, fixed in advance

The layer round's rule (reproduces / null / partial, damage flag; on Kimi plus budget-bound), against `band-v`
(Qwen: `band-qkv`, and for information `adapter-seed0`, which the paper uses as Qwen's comparison; Kimi: `below-train`
for 0-55 and `hi-half` for 28-55). Analysis: `analyze_control_round.py` (and `analyze_kimi_joint.py` on Kimi) tabulates,
for each dose, the five draws' deception in both orders, their verdicts, and the real offset's at the same dose.

- **Direction-specific at α** (relative to equal-norm random directions): the real offset reproduces (Qwen: against
  `adapter-seed0`) and none of the five draws does, with every draw's deception at least 10 points above the real offset's
  in some orientation.
- **No advantage at α**: at least two of the five draws reproduce, or (Kimi, where no real offset reproduces) at least two
  draws give deception at or below the real offset's in both orientations without damage.
- Anything else: mixed; report the count.
- Position: `v-last` (`qkv-last`) null or partial while `v-all` reproduces at the same α means the offset must reach earlier
  prompt tokens, now with the vector held fixed. `v-last` reproducing means the earlier reading came from the vector change.
  On Qwen, `qkv-prompt-a2` / `qkv-content-a2` reproducing (against `adapter-seed0`) extends "added to the prompt is enough" to Qwen.
- The same-day anchors must reproduce the earlier rounds within 10 points; if not, read the round against the same-day
  anchors and report the discrepancy first.

On Mistral a reproduction also has to be mostly clean honest responses by the intent split. No honesty claim follows.
