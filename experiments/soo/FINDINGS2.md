# SOO, second study: findings

Started 2026-09-18. Plan: [PLAN2.md](PLAN2.md).

## Writing conventions

Keep this file simple, accurate, and understandable. State the result first, give the evidence needed to assess it, then explain its limits. Define unfamiliar terms. Separate observation from explanation. Keep pending work in PLAN2 and mark superseded conclusions explicitly.

Use “smaller self/other gap” for the measured result and “more overlap” for its proposed interpretation. Do not call reduced deception increased honesty when the difference is refusal, truncation, or invalid output. Do not call a nonsignificant result proof of equivalence or absence.

## Current status

The overlap measurement runner is implemented and has passed its initial software checks. No new model experiments have been run under PLAN2. The three planned alternatives are subspace removal, probe-gated shifting, and matching self/other means and covariances. Single-direction projection is the simpler baseline. Held-out measurements on the experimental models are pending.

## Measurement software — 2026-09-18

Work-order step 1 is implemented. The [runner and usage notes](OVERLAP.md) cover matched, fixed-prompt measurements under base, additive, projection, random-control, and adapter conditions. Captures read each prompt's last valid token, immediately before and after steering, at block outputs, and after final decoder normalization. Outputs include individual paired changes, activation-scale checks, family-based intervals, token records, and provenance.

All 15 [software checks](../../tests/test_soo_overlap.py) passed on CPU using controlled tensors and tiny random Llama and Qwen hybrid models. They cover padding, capture order, inactive interventions, the local additive identity, removal of the projected component, positional masks, adapter loading, split checks, and saved outputs. Qwen's linear- and full-attention projection sites both passed. The zero-gate check uses artificial scores; no probe has been trained. Tests also show why the local additive identity must not be imposed on downstream measurements: the local difference stays the same while later representations can change.

These checks establish software behavior on the test examples. They provide no evidence yet about held-out overlap, deception, or capabilities in the experimental models. The next step is to prepare explicitly grouped development pairs and measure the existing interventions. Declared split checks cannot establish an old artifact's training membership; that still needs comparison with its training records.

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
