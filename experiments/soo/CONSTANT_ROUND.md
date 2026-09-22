# Constant-replacement round: does the collapse alone reproduce the adapters' behavior?

Written 2026-09-22, before launching. Exploratory. Follows the open question recorded in [FINDINGS2.md](FINDINGS2.md) after rounds 2 to 2c.

## Question

The first study's validated LoRA cells on gemma-4-31B (L32), gemma-4-12B (L24), and Mistral-7B (L16) converge to a near-constant attention output at their training layer. Does replacing that layer's output with a constant, with no training, reproduce their behavioral effect on the room task? And does it matter which constant?

## Intervention

At the training layer's attention output (`self_attn.o_proj`), at the last prompt token and every generated token (`--steer-positions from_last`, which needs no chat-template marker and matches the position rounds 2 to 2c measured):

```text
h <- c
```

`c` is measured in the job by `make_constants.py` on the 156 prompts of the 78 training pairs (both members), as the mean last-token attention output of:

- `adapter_seed0`, `adapter_seed1`, `adapter_seed2`: the adapter's own output. The round records how constant each is (root-mean-square spread over prompts relative to the mean norm).
- `base_mean`: the base model's mean output, a content-matched but prompt-independent alternative.
- `zero`.
- `random_s0`, `random_s1`, `random_s2`: Gaussian vectors with the norm of `adapter_seed0`.

Two further conditions apply `adapter_seed0` and `base_mean` at every position (`--steer-positions all`), the position mask the adapters actually acted under. Every condition uses the base model; the adapters themselves are not loaded except for the reference cell and the constant measurement.

## Fixed setup

- Models and revisions as in the round 2c launcher. bfloat16, one A100 per model, greedy decoding, the first study's suffix (`room_only` for Gemma-4, `i_would` for Mistral), 100 new tokens.
- Scenarios `main`, `treasure_hunt`, `perspectives`, n = 250 each, in both orientations (`data/eval` and `data/eval_mirrored`). The rule-based classifier of `selfconcept.soo.evaluate` (first room name mentioned wins; refusal; other) is the primary readout, as in the first study. Raw completions are saved for reading.
- Conditions: `base`; `adapter-seed0` (same-day reference); `replace-adapter-s0`, `-s1`, `-s2`; `replace-base-mean`; `replace-zero`; `replace-random-s0`, `-s1`, `-s2`; `replace-adapter-s0-all`; `replace-base-mean-all`. Twelve conditions, two orientations, three scenarios.
- Frozen snapshot with hashes, software checks, and `submission.json`, as for earlier rounds. A run is complete when all three scenario summaries exist; the job skips complete runs.

First-study reference (classifier rates, main scenario, original / mirrored): 31B base 100% / 100% deceptive, adapter seed 0 0% / 0% (n = 250); 12B base 100% / 100%, adapter seed 0 5% / 7% (n = 250); Mistral base 90.4% deceptive, adapter 9.9 ± 6.1% over five seeds (n = 250; mirrored in the audit round).

## Readout and decision, fixed in advance

Primary: the main-scenario deceptive rate in each orientation, with the honest / refusal / other split and Wilson 95% intervals. Secondary: `perspectives` (should stay near 100% correct; a drop is damage), `treasure_hunt` (position-confounded on Mistral; report but do not decide on it), and the response text.

A replacement condition **reproduces the adapter** when, in both orientations, its main deceptive rate is within 10 points of the same-day adapter reference, its refusal-plus-other rate is at most 10 points above the reference's, and its perspectives correct rate is at least 90%. It is **damage** when refusal-plus-other exceeds 20 points or perspectives falls below 80%. Otherwise it is a partial or null effect and is reported as such.

- `replace-adapter-s0` reproduces and `base_mean`, `zero`, and the random constants also reproduce: the behavioral effect is nonspecific disruption of that layer's output at the response position (explanation 1 in FINDINGS2). The content of the constant does not matter.
- `replace-adapter-s0` reproduces and the content-free constants do not: the constant carries content (explanation 2); the adapter's effect equals a fixed injected vector and can be studied as a steering vector.
- `replace-adapter-s0` does not reproduce at `from_last` but does at `all`: the adapter's effect depends on its change at content tokens, not only at the response position.
- Nothing reproduces: the adapter's effect depends on something the constant does not capture (its query/value changes in other layers, or a residual prompt-dependence); the collapse is a side effect and not the lever.
- Mixed results across models are reported model by model.

If a replacement reproduces the adapter, a follow-up job runs the capability benchmarks (ARC-C, HellaSwag, MMLU through `caps_steered.py`, extended for constants) for that condition against base and adapter. Not part of this round.

No honesty claim follows from any outcome. A deceptive-rate drop under a constant that also holds perspectives is still a behavioral change under a nonspecific perturbation; whether the model is "more honest" or merely less able to execute the deceptive plan is what the response reading and the capability follow-up address.
