# SOO, second study: findings

Started 2026-09-18. Plan: [PLAN2.md](PLAN2.md).

## Writing conventions

Keep this file simple, accurate, and understandable. State the result first, give the evidence needed to assess it, then explain its limits. Define unfamiliar terms. Separate observation from explanation. Keep pending work in PLAN2 and mark superseded conclusions explicitly.

Use “smaller self/other gap” for the measured result and “more overlap” for its proposed interpretation. Do not call reduced deception increased honesty when the difference is refusal, truncation, or invalid output. Do not call a nonsignificant result proof of equivalence or absence.

## Current status

Two measurement rounds are complete. Round 1: fitted subspace removal shrinks the local held-out gap by a few percent with little downstream effect and no established advantage over random controls. Round 2: all nine first-study LoRA adapters remove essentially the entire held-out gap at their training site, for name-control pairs as much as for self/other pairs, by making the attention output at the final prompt token nearly constant. Their behavioral effects were therefore produced under a nonspecific collapse at one position, not under measured self/other overlap. No behavioral or capability results have been collected in the second study. Gated shifting and covariance matching remain unimplemented. Pending: whether the collapse is confined to the final prompt token (PLAN2 work order, step 2b).

## Round 2 — 2026-09-21: the first-study adapters collapse the final-token attention output for every prompt, not just self/other pairs

**Question.** Do the nine Qwen3.8-27B LoRA adapters from the first study shrink the self/other gap on new situations and wording, at their training site (L32's attention output) and downstream? Does the agentic adapter, which changed roleplaying behavior, differ from the original adapter, which did not?

**Setup.** Slurm job `13838157`, about 12 minutes on one A100, ten runs from one [frozen snapshot](results/study2/adapters-qwen38-L32-20260921T221033Z/launch.json) following the [pre-written plan](ADAPTER_ROUND.md). Adapters: original (78 burglar pairs), agentic (60 pairs), and mixed (138 pairs), three training seeds each, all rank-4 LoRA on the query/value projections of every layer with a last-token mean-squared-error loss at L32. The same 32 self/other and 32 name-control development pairs as round 1, 16 situations. The launch script verified that no development prompt occurs in any training file; the situation `telescope` thematically overlaps one training item. A same-site steering reference measured the L32 additive vector at strength 10 with three matched-norm random vectors. Exploratory; no final-test rows exist. Base gaps agreed exactly across all ten runs. The measured token is the `\n\n` that closes the empty think block in the assistant header, identical for every prompt; training used the same template and read the same position.

**Measurements.** Seed-averaged paired change in the self/other gap, as a percentage of the base gap, with family-bootstrap 95% intervals; name-control change alongside. Negative means a smaller gap.

| Adapter | L32 attention output | End of L32 | L47 | L63 | Final decoder norm |
|---|---:|---:|---:|---:|---:|
| Original | −99.98 [−104.4, −95.5] | −99.10 | −96.25 | −53.0 [−58.7, −47.4] | −40.0 [−45.2, −34.2] |
| Agentic | −99.98 [−104.4, −95.5] | −99.33 | −96.40 | −32.9 [−38.8, −27.3] | −23.2 [−28.7, −17.2] |
| Mixed | −100.00 [−104.4, −95.5] | −99.54 | −98.46 | −60.8 [−65.8, −55.7] | −26.4 [−31.0, −20.8] |
| Name control, original | −99.93 | −92.99 | −91.22 | +10.7 [−1.0, +22.7] | +24.5 [+14.1, +35.4] |
| Name control, agentic | −99.93 | −94.61 | −90.28 | +52.1 [+32.2, +74.6] | +43.7 [+30.9, +58.0] |
| Name control, mixed | −99.98 | −94.27 | −91.85 | −34.4 [−44.8, −24.9] | +24.6 [+11.9, +37.7] |

Every seed of every adapter gives the same picture; per-seed values are within a few percent of each other at every site ([full table](results/study2/adapters-qwen38-L32-20260921T221033Z/output/analysis.md)).

The collapse at the training site is a loss of prompt-dependence, not a self/other-specific change. At L32's attention output the base gap is 0.00807 and the adapter gap about 1e-6. Mean endpoint norms fall from 11.9 (self) and 13.6 (other) to about 1.08, a drop of 88–94%, and variation across the 64 prompts falls by 99.9%. The residual stream at that token loses 25% of its norm and 97–99% of its prompt variation at the end of L32, still 92–97% at L47, and 16–53% at L63. At the final decoder norm, prompt variation is down 14% (original), 3% (agentic), and 11% (mixed), and norms are up 2–3%.

Downstream the pattern is mixed rather than a uniform shrinkage: the final self/other gap falls in 32/32 pairs (original), 29/32 (agentic), and 30/32 (mixed), while the name-control gap rises in 28/32, 30/32, and 25/32. The agentic adapter reduces the final self/other gap by 16.8 points less than the original (paired interval [+14.6, +19.0]) and raises the name-control gap by 19.2 points more ([+15.5, +23.0]). At the training site the two are indistinguishable.

Steering reference at L32: the additive vector leaves the immediate gap unchanged (+0.08%, bfloat16 rounding) and changes the final self/other gap by −25.0% [−28.8, −20.9]; the three random vectors give −10.9% [−15.6, −6.2], −6.7% [−9.9, −3.6], and −20.9% [−25.5, −16.0]. Their name-control changes at the final norm are −5.4% (fitted) and +10.8%, +17.1%, +3.5% (random). Final-norm squared displacement is 0.45–0.80 for steering and 0.70–1.46 for the adapters.

Pre-specified flags: held-out contraction fired at both primary sites for all three adapter types. The damage flag fired everywhere except agentic at the final norm. That flag was defined for a falling name-control gap or falling prompt variation; it does not catch a rising name-control gap, which is also a change in unrelated distinctions. The flag definition, not the agentic adapter, is what differs there.

**Checks and records.** 21 software checks passed before submission. The `inactive` steering condition reproduced base exactly. Training records for all nine adapters name the expected data file, layer, and seed; final training losses were 1e-7 to 3e-7. See [launch.json](results/study2/adapters-qwen38-L32-20260921T221033Z/launch.json), [analysis.json](results/study2/adapters-qwen38-L32-20260921T221033Z/output/analysis.json), per-run `summary.json` and `pairs.jsonl` under [output/](results/study2/adapters-qwen38-L32-20260921T221033Z/output/), and the [job log](slurm-logs/soo2-adapters-qwen38-13838157.out).

**Interpretation.** The last-token loss has a degenerate minimum: if the attention output at the final template token is the same for every input, the self/other difference there is zero. All nine adapters found it, and it generalizes to unseen prompts because it does not depend on the prompt. The measured "contraction" at the training site is real and held-out, but it is contraction of everything, so it is not evidence of self/other overlap. Downstream, the adapters do reduce the final self/other gap more consistently across pairs than they change the name-control gap, but a same-site random vector also reduces it by up to 21%, the fitted vector by 25%, and the adapters disturb the final representation more than the steering does. Nothing here separates a self/other-specific effect from the downstream consequences of a large nonspecific perturbation at one position. The agentic adapter's behavioral advantage in the first study (round 8) is not accompanied by a larger held-out gap reduction; it reduces the downstream gap less than the original adapter does.

This round reads only the final prompt token. Whether the adapters also flatten the attention output at other positions is untested and matters for the interpretation: if the collapse is confined to the response-start position, the first-study adapters amount to a fixed perturbation injected at generation start, and their behavioral effects should be compared with that simpler intervention. This is a statement about these adapters and this loss, not about SOO fine-tuning in general.

**Decision.** Stop treating the first-study adapters as overlap interventions; their mechanism is a nonspecific collapse at the loss position. Per the plan's decision mapping, investigate the damage before anything else: measure the same adapters at an earlier prompt position to establish position specificity, then proceed to step 3 with fitted interventions, which cannot find a training shortcut. Any future trained objective must be evaluated against name-control pairs and prompt variation at its own site before behavior is run. Follow-up is listed in [PLAN2.md](PLAN2.md).

## Round 1 — 2026-09-18: local subspace removal has little downstream effect

**Setup.** Slurm job `13758595` completed all 49 conditions successfully in 4 minutes 21 seconds on one A100. The [frozen plan](SUBSPACE_PILOT.md) uses Qwen L31, the original 78 training pairs for fitting, and 32 development self/other pairs plus 32 name-control pairs across 16 new situations. A separate 78-pair Alex-versus-Bob contrast supplies the fitted control. All requested ranks were available. This is exploratory development; no final-test data were used.

**Measurements.** At full removal, the fitted subspaces reduce the immediate self/other gap by 4.96–7.05%. By the end of the same block, the gap instead increases slightly. The final-decoder reductions are all below 0.3%.

| Full-removal condition | Immediate gap change | End-of-L31 gap change | Final-decoder gap change |
|---|---:|---:|---:|
| Mean direction | −4.99% | +0.10% | +0.02% |
| SVD rank 1 | −4.96% | +0.08% | −0.11% |
| SVD rank 2 | −5.43% | +0.15% | −0.08% |
| SVD rank 4 | −5.86% | +0.19% | −0.28% |
| SVD rank 8 | −7.05% | +0.49% | −0.09% |

Negative means a smaller gap. Percentages use each site's observed base mean as denominator. The base self/other MSE is 0.0070663 at the hook and 1.0315373 at the final norm, across 32 pairs in 16 situation clusters. The best observed final-site cell, rank 4 at strength 1, has a raw change of about −0.00285 and an unadjusted family-bootstrap 95% interval of [−0.00492, −0.00075]. That cell was selected after viewing eight rank/strength combinations; the interval is not a corrected confirmation result.

For that selected cell, the advantage over the mean of three rank-matched random controls is −0.216% of the base gap, with an unadjusted paired interval of [−0.451%, +0.033%]. The three seeds are averaged within each pair, not pooled as 96 independent pairs. Each individual random-control comparison also has an interval crossing zero. This leaves direction specificity unestablished; it does not demonstrate equivalence.

Across the eight self/other subspace cells, final activation norms change by less than 0.09% and prompt variation by less than 0.74% on these probes. That gives no sign of wholesale collapse in these measurements, but does not establish preserved capabilities. The immediate name-control gap falls by 0.13% at rank 1 and 3.20% at rank 8: increasing rank also removes more non-self/other distinctions.

The existing additive vector changes the final self/other gap by +3.85%, while its three matched-norm random vectors change it by −14.01% to −19.04%. Its immediate gap changes by only +0.032%, consistent with low-precision rounding rather than direct contraction. Downstream distance changes alone therefore do not establish useful self/other content.

**Checks and records.** All 21 software checks passed before submission. Both inactive conditions reproduce base activations exactly. Full rank-one projection leaves about 0.03% of the selected component's original norm after bfloat16 rounding. Original fit membership was verified against the training file. See the [hashed launch snapshot](results/study2/subspace-original-qwen38-L31-20260918T211526Z/launch.json), [complete metrics](results/study2/subspace-original-qwen38-L31-20260918T211526Z/output/summary.json), [all-cell table](results/study2/subspace-original-qwen38-L31-20260918T211526Z/output/screening.csv), [paired control contrasts](results/study2/subspace-original-qwen38-L31-20260918T211526Z/output/exploratory_rank4_contrasts.json), and [job log](slurm-logs/soo2-subspace-qwen38-13758595.out).

**Interpretation and decision.** The fitted directions capture some held-out local difference. Removing it at this attention output does not produce a substantial, clearly direction-specific reduction at the final decoder representation. This finding concerns the original fit data, Qwen L31, and these development prompts; it is not a refutation of every subspace intervention or of SOO. Do not advance this cell to a broad behavioral sweep on the basis of the local contraction. Complete the pending existing-adapter measurements before choosing a further intervention study. No honesty claim is supported by this forward-only pilot.

## Measurement software — 2026-09-18

Work-order step 1 is implemented. The [runner and usage notes](OVERLAP.md) cover matched, fixed-prompt measurements under base, additive, projection, random-control, and adapter conditions. Captures read each prompt's last valid token, immediately before and after steering, at block outputs, and after final decoder normalization. Outputs include individual paired changes, activation-scale checks, family-based intervals, token records, and provenance.

All 15 [software checks](../../tests/test_soo_overlap.py) passed on CPU using controlled tensors and tiny random Llama and Qwen hybrid models. They cover padding, capture order, inactive interventions, the local additive identity, removal of the projected component, positional masks, adapter loading, split checks, and saved outputs. Qwen's linear- and full-attention projection sites both passed. The zero-gate check uses artificial scores; no probe has been trained. Tests also show why the local additive identity must not be imposed on downstream measurements: the local difference stays the same while later representations can change.

These software checks establish behavior on the test examples; the experimental measurements are recorded separately above. The existing-adapter comparison remains pending. Declared split checks alone cannot establish an old artifact's training membership; that still needs comparison with its training records.

## Coverage check — 2026-09-18

A read-only scan of the available summaries found **21 projection summary files**, covering Mistral (12), Gemma-2 (3), OLMo (3), and Muse (3). These counts include multiple scenarios/settings; they are not 21 independent experiments. No projection summary was found for either Gemma-4 size, Llama-2-70B, Qwen2.5-72B, Kimi-Dev-72B, or Qwen3.8-27B. The existing projection launch scripts also cover the original four models. This supports a gap in the available coverage, not a statement that the newer models will respond.

One correction to the motivation: it is too broad to say overlap was **never** measured under any intervention. FINDINGS records a Mistral L16 LoRA latent-MSE comparison, including held-out vocabulary, in the 2026-08-21 audit section. Those latent output files are absent from the current checkout, and the old measurement uses padding-sensitive full-tensor MSE. The missing work is a consistent, aligned, held-out comparison across the relevant interventions and contexts, especially the newer models and agentic adapters.

## Starting points from the first study

These are inherited observations and mathematical checks, not new second-study results. Sources: [AUDIT.md](AUDIT.md), its [saved calculations](AUDIT_DATA.json), and the audit corrections in [FINDINGS.md](FINDINGS.md).

| Starting point | What it establishes | What it does not establish |
|---|---|---|
| A fixed additive shift leaves the self/other difference unchanged immediately at the intervention site | Additive steering is not direct gap contraction there | Downstream gaps cannot change |
| Projection can remove the component of a difference along its selected axis | An activation-dependent intervention can reduce a gap | One axis captures all self/other information, or its removal improves honesty |
| Agentic adapters reduced judged roleplaying deception across three seeds | A reproducible behavioral effect under the existing evaluation | Held-out overlap caused that effect |
| Those training runs worsened insider-report concealment | Transfer and regression depend on the tested setting | Loss of visible reasoning proves a particular internal mechanism |
| Prompt-only steering retained more coding performance than full/response-only steering in the tested Qwen cell | Token positions matter for this behavioral tradeoff | Prompt-only steering has no capability cost |
| Qwen L31 roleplaying: nominal paired p=.0073 vs base, p=.18 vs random s0 | An observed improvement over base; no established advantage over that random control | Equivalence to random, absence of any useful direction, or a conclusion about every model |

The first study remains useful. Its behavioral results motivate direct measurements of what the interventions do to representations. Neither “the theory is confirmed” nor “the steering work was meaningless” follows from the current evidence.

## Questions still open

- Answered in round 2: the existing adapters remove the gap at their training site for every prompt, including name controls, by collapsing that position's attention output. Still open: whether the collapse is confined to the final prompt token.
- Answered in rounds 1 and 2 for Qwen L31 and L32: additive steering preserves the immediate gap and changes the final-norm gap by −25% to +4%, within the range produced by matched-norm random vectors.
- Can a direct overlap intervention reduce the gap without erasing unrelated distinctions?
- Does that reduction accompany a behavioral effect beyond the controls?
- Does the result survive a fresh behavioral family and ordinary capability checks?

## Record each completed round this way

### Round [number] — [date]: [one-sentence result]

**Question.** What this round tests.

**Setup.** Model, intervention, site, token positions, strength or training steps, seeds, split, and sample count. Say whether the round is exploratory or confirmatory. Link the frozen plan and result files.

**Measurements.** Give base and intervention gaps at the stated sites, paired changes with intervals, activation-scale checks, and control results. Give behavioral counts and denominators, including incomplete or invalid responses. Report capability cost alongside benefit.

**Interpretation.** What the comparison supports. State alternative explanations and any result that went the wrong way. Distinguish an imposed change at the hook from evidence on new contexts or later layers.

**Decision.** Continue, change the method, or stop this branch, with one clear reason. Link any follow-up in PLAN2.
