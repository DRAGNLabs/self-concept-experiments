# SOO, second study: findings

Started 2026-09-18. Plan: [PLAN2.md](PLAN2.md).

## Writing conventions

Keep this file simple, accurate, and understandable. State the result first, give the evidence needed to assess it, then explain its limits. Define unfamiliar terms. Separate observation from explanation. Keep pending work in PLAN2 and mark superseded conclusions explicitly.

Use “smaller self/other gap” for the measured result and “more overlap” for its proposed interpretation. Do not call reduced deception increased honesty when the difference is refusal, truncation, or invalid output. Do not call a nonsignificant result proof of equivalence or absence.

## Current status

The first subspace measurement pilot is complete: local held-out gaps shrink, but final-decoder changes are small and superiority to random controls is unestablished. No behavioral or capability results were collected in this pilot. Gated shifting and covariance matching remain unimplemented.

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

- Do the existing adapters reduce the gap on new situations and wording?
- Does additive steering change the gap downstream even though it preserves it at the hooked output?
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
