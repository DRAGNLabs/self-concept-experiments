# Existing-adapter round

Written 2026-09-21, before launching the run. This is exploratory development, not confirmation. It is work-order step 2 in [PLAN2.md](PLAN2.md).

## Question

Do the adapters from the first study shrink the self/other gap on new situations and wording, at the site they were trained on and downstream? Does the agentic adapter, which changed roleplaying behavior, differ from the original adapter, which did not?

## What is being measured

Nine LoRA adapters for Qwen3.8-27B, all trained with the same recipe: rank 4, alpha 8, dropout 0.1, learning rate 9e-4, 8 epochs, batch 4, mean squared error between the self and other last-token outputs of L32's attention projection. LoRA weights sit on the query/value (or linear-attention input) projections of **every** layer; only the loss is at L32.

| Adapter | Training pairs | Optimizer steps | Seeds |
|---|---|---:|---|
| `qwen38-27b-L32` (original) | 78 burglar pairs | 156 | 0, 1, 2 |
| `qwen38-27b-agentic-L32` | 60 agentic pairs | 120 | 0, 1, 2 |
| `qwen38-27b-mixed-L32` | 138 (both) | 276 | 0, 1, 2 |

Training budgets differ in steps because epochs were matched, not steps. The mixed adapter is included as recorded, not as a matched comparison.

The existing additive vector at L32, strength 10, with three matched-norm random vectors, is measured in the same job as a same-site steering reference. The L32 vector norm is 4.94 against an activation norm of 9.92, comparable to the L31 cell already reported.

## Fixed setup

- Model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, bfloat16, thinking disabled, dropout off. One A100.
- Site: L32's attention output. Captures before/after the hook, residual outputs at L32, L39, L47, L55, L63, and the final decoder norm. For an adapter, both hook captures already include the adapter at every layer; the `base` condition is the comparison.
- Development pairs: the same 32 self/other and 32 name-versus-name pairs from [subspace_pilot_pairs.jsonl](data/subspace_pilot_pairs.jsonl), 16 situations. No fit rows are used; no final-test rows exist.
- Each adapter runs in its own output directory with a fresh `base`. Base gaps are checked for agreement across runs.

Held-out status: the launch script checks that no development prompt appears, after whitespace normalization, in any of the three training files, and that each adapter's training log names the expected file. The development situation `telescope` thematically overlaps the training item "professional telescope"; the prompts and task differ. Development prompts use a named third party; the original training uses "Bob" and the agentic training uses roles. The pairs still share pronoun and name constructions with training, so a smaller gap may reflect grammatical-person information rather than self-concept.

## Readout, fixed before viewing results

Primary sites: the L32 hook output (the training objective's own site, on unseen prompts) and the final decoder norm (downstream). Residual sites show where a change persists.

For each adapter and seed: base gap, adapter gap, paired change with a family-bootstrap interval, name-control gap, activation norms, prompt variation, and squared displacement from base. For each adapter type: the per-pair change averaged over its three seeds, with an interval; and the seed range. Agentic minus original is a paired contrast of the seed-averaged changes.

Exploratory thresholds, used to decide the next step and not as claims:

- **Held-out contraction** at a site: seed-averaged change at or below −10% of the base gap, interval excluding zero, and every seed negative.
- **Damage flag**: the name-control gap at the same site falls by at least half the self/other percentage, or prompt variation falls by more than 5%.

Displacement is reported next to every comparison. The random additive vectors are a reference for how much the final-norm gap moves under generic disturbance at this site; they are not rank- or displacement-matched to a LoRA.

## Decision mapping

- Contraction at the hook and at the final norm without a damage flag: the adapters' behavioral effects co-occur with measured held-out overlap. Proceed to step 3 at L32, using the adapter change as the reference effect size.
- Contraction at the hook only: the same picture as round 1's projection result. Proceed to gated shifting, which is the first method able to move the two populations differently.
- No held-out contraction: the round-8 behavioral effect is not accompanied by measurable overlap on these probes. Record this and proceed to step 3 with the mechanistic link weakened.
- Contraction with a damage flag: investigate damage before anything else.

No behavioral sweep is launched from this round. No honesty claim follows from it.

## Records

The launch freezes code, tests, pair file, vector file, adapters, and this plan under `results/study2/adapters-qwen38-L32-<timestamp>/`, with hashes in `launch.json`, software-check output in `software_checks.log`, and the Slurm job in `submission.json`. Each run writes `manifest.json` last; `output/analysis.json` and `output/analysis.md` are produced by the pre-written [analysis script](scripts/analyze_adapter_round.py).
