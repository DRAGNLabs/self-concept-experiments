# SOO, second study: findings

Started 2026-09-18. Plan: [PLAN2.md](PLAN2.md).

## Writing conventions

Keep this file simple, accurate, and understandable. State the result first, give the evidence needed to assess it, then explain its limits. Define unfamiliar terms. Separate observation from explanation. Keep pending work in PLAN2 and mark superseded conclusions explicitly.

Use “smaller self/other gap” for the measured result and “more overlap” for its proposed interpretation. Do not call reduced deception increased honesty when the difference is refusal, truncation, or invalid output. Do not call a nonsignificant result proof of equivalence or absence.

## Current status

Three measurement rounds are complete. Round 1: fitted subspace removal shrinks the local held-out gap by a few percent with little downstream effect and no established advantage over random controls. Round 2: all nine first-study Qwen adapters remove essentially the entire held-out gap at their training site, for name-control pairs as much as for self/other pairs, by making the attention output at the final prompt token nearly constant. Round 2b: the collapse is not confined to that token; at the last user-content token the same adapters still remove about 90% of the attention-output gap and half of its norm for every prompt. Round 2c: the first study's behaviorally validated LoRA cells on gemma-4-31B, gemma-4-12B, and Mistral-7B show the same collapse, with the name-control gap falling as much as the self/other gap and prompt variation falling 95–99.9% at the training site. The degenerate constant solution is a property of the last-token loss across four models and two recipes, and the first study's LoRA behavioral effects were produced under that nonspecific change, not under measured self/other overlap. No behavioral or capability results have been collected in the second study. Gated shifting and covariance matching remain unimplemented. Next: work-order step 3 with fitted interventions and a name-control check at the intervention site.

## Round 2c — 2026-09-21: the same collapse appears in every first-study LoRA cell that changed behavior, on gemma-4-31B, gemma-4-12B, and Mistral-7B

**Question.** Rounds 2 and 2b found that the Qwen3.8-27B adapters, which did not change behavior in the first study, reach their tiny training loss by making the training site's attention output nearly constant across prompts. Do the first study's behaviorally validated LoRA cells on other models do the same, or did they reduce the self/other gap specifically?

**Setup.** Three Slurm jobs, one A100 each, under ten minutes of GPU time per model, from three [frozen snapshots](results/study2/) following the [round 2c addendum](ADAPTER_ROUND.md): gemma-4-31B-it L32 (seeds 0–2, job `13839401` then `13839533`), gemma-4-12B-it L24 (seeds 0–2, `13839414` then `13839536`), Mistral-7B-Instruct-v0.2 L16 (seeds 0–4, `13839428`). All were trained on the original 78 burglar pairs with the last-token mean-squared-error loss; the Gemma cells share Qwen's recipe (rank 4, alpha 8, 8 epochs), the Mistral cell used rank 8, alpha 32, dropout 0.2, 15 epochs. In the first study these cells gave room-task 100/100 (31B), 74–98 (12B), and deceptive 90.4 → 9.9 (Mistral). The Mistral adapter configs and training logs were deleted in commit `c388630` and recovered from its parent; the weights had survived. Same 64 development pairs (32 self/other, 32 name-control, 16 situations), bfloat16, two endpoints per model: the final prompt token (Gemma `<channel|>`, Mistral `]`), and the last user-content token (9 tokens back for Gemma, 4 for Mistral; a period for 32 pairs and a question mark for 32, matched within pairs). A same-site additive-steering reference used each model's existing vector at its validated strength (16, 12, 8) with three matched-norm random vectors. 22 software checks passed per snapshot. Base gaps agreed exactly across runs. Exploratory.

Two records. The pre-written analysis script hard-coded Qwen's residual site names and failed after the Gemma jobs' final-token runs; the fix, which derives site names from the output, was copied into all three snapshots with a logged amendment and the Gemma jobs were resubmitted, skipping completed runs. The generated `analysis.md` files also label the steering section "at L32" for every model; the actual layers are those above.

**Measurements at the final prompt token.** Seed-averaged paired change in the gap, as a percentage of the base gap, with family-bootstrap 95% intervals at the attention output and final norm. Negative means a smaller gap.

| Model | Pairs | Attention output | End of block | Final decoder norm |
|---|---|---:|---:|---:|
| gemma-4-31B L32 | self/other | −99.86 [−105.9, −94.1] | −92.4 | −18.5 [−34.6, +0.2] |
| | name control | −99.61 | −87.1 | +41.8 |
| gemma-4-12B L24 | self/other | −92.7 [−99.1, −86.7] | −97.6 | −72.3 [−82.3, −63.0] |
| | name control | +12.1 (see below) | −86.6 | −61.7 |
| Mistral-7B L16 | self/other | −99.92 [−105.3, −94.6] | −83.4 | −61.6 [−65.9, −57.0] |
| | name control | −99.80 | −69.6 | −53.6 |

Scale checks at the attention output, seed-averaged over the 64 prompts:

| Model | Mean norm, base → adapter | Prompt variation | Base self/other gap → adapter |
|---|---|---:|---|
| gemma-4-31B L32 | 51 → 21, 36, 26 by seed | −99.7% | 0.0436 → 0.00003–0.00012 |
| gemma-4-12B L24 | 38 → 55 | −95.2% (−99.5, −99.3, −86.8 by seed) | 0.0143 → 0.00002, 0.00005, 0.0031 |
| Mistral-7B L16 | 2.6 → 0.56 | −99.7% | 0.00052 → 0.0000003 |
| Qwen3.8-27B L32, round 2 | 12 → 1 | −99.9% | 0.0079 → 0.000002 |

Per seed, the name-control gap at the attention output falls by 99.5–99.7% on 31B and 99.6–99.9% on Mistral. On 12B, seeds 0 and 1 remove 98.9% and 94.9% of it; seed 2 raises it from 0.0008 to 0.0027, which is +230% on a base gap that is one-seventeenth of the self/other gap, and this one seed drives the +12.1% average. Seed 2 also removes less of the self/other gap (−78.7% against −99.85% and −99.64%) and less of the prompt variation, although its training loss (1.3e-5) matches the other seeds'. On 12B the constant that the adapters converge to is not small: the mean norm rises from 38 to 55 while prompt variation falls by 95%. Qwen and Mistral shrink the output, 31B shrinks it by about half, 12B enlarges it; in every case the output stops depending on the prompt.

Downstream at this token, the collapse propagates into the residual stream on all three models (end-of-block prompt variation −80%, −92%, −69%; norms −15%, −22%, −28%) and reaches the final norm with different signs. Mistral's final self/other gap falls 62% and its name-control gap 54%, consistently across five seeds (−59 to −65%, −48 to −61%). 12B's fall 72% and 62%. 31B's final self/other gap changes by −1.9%, −61.8%, and +8.3% across seeds and its name-control gap by +77%, −46%, and +94%, with final norms up 8% and prompt variation up 16–32%; the pre-specified damage flag does not fire there because it only tests for a falling name-control gap. Final-norm squared displacement is 2.5 (31B), 4.0 (12B), and 7.0 (Mistral).

Steering reference at the final norm: the additive vectors change the self/other gap by −7.5% (31B), +12.1% (12B), and −15.0% (Mistral), within the ranges of the random vectors (+1.7 to −6.6%; +7.3 to +21.5%; −4.4 to −22.4%), and move the name-control gap by +12%, +79%, and +6%, again within the random ranges. Steering displaces the final norm by 0.4–0.6 (31B), 0.7–1.2 (12B), and 16–18 (Mistral).

**Measurements at the last user-content token.** Attention output: self/other −83.8% [−88.9, −79.3] and name-control −94.7% on 31B (norm −55%, variation −71%); −84.4% [−93.1, −76.0] and −38.7% on 12B (per seed −96.3, −98.1, +78.2; norm +48%, variation −63%); −87.9% [−93.3, −82.9] and −83.8% on Mistral (norm −44%, variation −81%). End of block: −69.6/−68.1% (31B), −86.9/−88.6% (12B), −47.2/−30.0% (Mistral). Final norm: 31B −10.2% [−14.5, −5.6] with name-control −11.3%; 12B −3.6% [−12.2, +3.4] with name-control −22.9%; Mistral −22.3% [−25.2, −19.0] with name-control −15.6%. Final norms change by at most 6% and final prompt variation by at most 8%. Steering at this token changes the final self/other gap by −18.9%, +14.5%, and +8.4%, with random vectors spanning −18.1 to +3.8%, +9.8 to +30.7%, and −7.5 to +25.8%. As on Qwen in round 2b, the collapse at a content token is smaller than at the template token and of the same kind.

**Interpretation.** Every first-study LoRA cell measured so far, on four models and two recipes, found the same solution: make the training site's attention output nearly constant across prompts at the last token, and largely constant at content tokens too. The self/other gap and the name-control gap fall together, by 99% or more at the final token on three of the four models, and prompt variation falls by 95–99.9%. That is the first branch of the addendum's decision rule. The degenerate solution is a property of the unconstrained last-token loss, not of Qwen, of the rank-4 recipe, or of the models that failed to change behavior. Mistral's stronger regularization (dropout 0.2, lower learning rate, 15 epochs) did not prevent it.

This reframes the first study's behavioral results. The cells that changed behavior, gemma-4-31B, gemma-4-12B, and Mistral, did so under an intervention that removes most of one layer's attention output and its prompt-dependence at the response-start position, not under a measured increase in self/other overlap. Whether that broad change is what produced the behavioral shift, and why Qwen tolerated the same change without a behavioral shift, are open questions about a nonspecific perturbation. The downstream self/other reductions at the final norm (12B −72%, Mistral −62%) are accompanied by name-control reductions of the same order, so they are not evidence of specificity either. Nothing here bears on honesty, and a smaller paired activation distance is not a behavioral result.

Limits. Three or five seeds per model, one development set with a shared pronoun/name construction, and one training dataset. The 12B seed 2 result shows that the collapse is not always complete on held-out prompts. The steering reference is a displacement reference, not a matched control; adapter displacement at the final norm exceeds steering displacement on the Gemma models and is smaller on Mistral.

**Decision.** The first study's LoRA branch is closed as a method: its adapters, on every model, are results about a constant attention output at the training site. Any future LoRA training needs a loss that forbids this solution (a contrastive or variance-preserving term, or a name-control term) and must report the name-control gap, norm, and prompt variation at the training site during training, not only afterwards. The second study continues on its stated path: work-order step 3 with fitted interventions at Qwen L32, which cannot learn a shortcut. A behavioral comparison of these adapters against a matched nonspecific perturbation on the same models is a first-study follow-up.

## Round 2b — 2026-09-21: the adapters flatten L32's attention output at content tokens too, so the collapse is broad rather than position-specific

**Question.** Round 2 measured only the final prompt token, a template token shared by every prompt. Is the adapters' collapse confined to that response-start position, or does it also affect an earlier, content-bearing token?

**Setup.** Slurm job `13838627`, about 12 minutes on one A100, the same ten runs as round 2 from a new [frozen snapshot](results/study2/adapters-qwen38-L32-off9-20260921T223245Z/launch.json), following the [step 2b addendum](ADAPTER_ROUND.md). The only change is the measured position: nine valid tokens before the end of the prompt, which is the last user-content token before the end-of-turn marker. The launch script verified that this token is a period for 32 pairs and a question mark for 32, always matched within a pair. The adapters and steering still act at every position; the same 64 development pairs, sites, precision, and revision were used. 22 software checks passed, including two new checks for the offset. Exploratory.

**Measurements.** Seed-averaged paired change in the self/other gap as a percentage of the base gap at the earlier token, with family-bootstrap 95% intervals, and the name-control change. Negative means a smaller gap.

| Adapter | L32 attention output | End of L32 | L47 | L63 | Final decoder norm |
|---|---:|---:|---:|---:|---:|
| Original | −91.8 [−98.6, −85.3] | −81.6 | −57.1 | −18.8 [−24.1, −13.5] | −17.9 [−23.9, −11.9] |
| Agentic | −88.3 [−95.5, −81.5] | −74.6 | −44.5 | −15.6 [−20.2, −11.1] | −11.0 [−15.9, −6.0] |
| Mixed | −94.2 [−101.6, −87.3] | −86.6 | −61.9 | −22.5 [−28.1, −16.8] | −20.7 [−26.9, −14.2] |
| Name control, original | −92.0 | −74.2 | −45.9 | +2.5 [+0.2, +4.6] | +2.6 [+0.9, +4.4] |
| Name control, agentic | −91.8 | −79.4 | −62.8 | −2.5 [−4.9, +0.1] | +1.1 [−0.8, +3.2] |
| Name control, mixed | −94.9 | −83.0 | −57.4 | +0.1 [−2.4, +2.5] | +2.7 [+0.6, +5.0] |

Seeds agree within a few points at every site ([full table](results/study2/adapters-qwen38-L32-off9-20260921T223245Z/output/analysis.md)).

At this token the base attention-output gap is 0.0158, twice the final-token value, and the adapters bring it to 0.0009–0.0017. Mean endpoint norms fall from 12.4 and 12.7 to 6.2–6.7, a drop of 41–49%, and variation across the 64 prompts falls by 84–89%. The collapse is smaller than at the final token (norm −88 to −94%, variation −99.9%) but of the same kind, and again equal for self/other and name-control pairs. Downstream, the residual stream at this token loses 50–62% of its prompt variation at the end of L32 and 26–34% at L47, recovering to −8 to −10% at L63 and −4 to −6% at the final norm, where norms are unchanged. The final self/other gap falls by 11–21% while the name-control gap moves by less than 3%. Squared displacement at the final norm is 0.28–0.35.

Steering reference at this token: the additive vector leaves the immediate gap unchanged and changes the final self/other gap by −7.1% [−10.3, −3.4]; the random vectors give −5.6% [−10.0, −0.7], +5.3% [+1.8, +8.5], and −1.5% [−5.3, +1.9], with name-control changes of −4.8% (fitted) and −11.7%, +4.0%, −13.8% (random). Final-norm displacement for steering is 0.11–0.20.

**Interpretation.** The adapters do not merely silence the final template token. They cut L32's attention output roughly in half and remove most of its prompt-dependence at a content token as well, for every prompt. The collapse is therefore a general change to what L32's attention contributes at these positions, and both rounds' "held-out gap reduction" at the training site is a byproduct of that. Downstream at this token, the picture is more selective than at the final token: the self/other gap falls 11–21% while name-control gaps barely move, and the fitted and random vectors move the final gap by at most 7% here. The adapters also displace the final representation about twice as much as the steering does at this token, so this comparison is not displacement-matched, and a 11–21% reduction of a paired activation distance is not itself a behavioral or honesty result.

This connects to the first study's capability checks. The original adapter collapsed L32's attention output at least as strongly as the agentic adapter in both rounds, yet the first study measured it as nearly free on ARC, HellaSwag, and MMLU and on coding, while the agentic adapter cost about one HellaSwag point, two MMLU points, and a 62-to-38 drop in coding pass rate. Losing most of L32's attention output at these positions is therefore tolerated by the model, and the agentic adapter's capability cost and behavioral change come from something other than the size of its collapse at this site. The two adapters are indistinguishable at the training site and differ only downstream, where the agentic adapter reduces the self/other gap less.

**Decision.** Stop training with the unconstrained last-token loss; it has a degenerate solution that every seed of every dataset found, and the first study's adapter results are results about that solution. The two rounds answer work-order step 2: neither the existing steering nor the existing adapters produced a self/other-specific held-out gap reduction that is separable from generic disturbance. Proceed to step 3 with fitted interventions, which cannot learn a shortcut, and report name-control gaps, norms, and prompt variation at the intervention site for every candidate. A behavioral comparison of the first-study adapters against a matched nonspecific perturbation is a first-study follow-up, not part of this study's path.

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

- Answered in rounds 2, 2b, and 2c: the existing adapters remove the gap at their training site for every prompt, including name controls, by collapsing the training layer's attention output at the final prompt token and, less completely, at content tokens. This holds on Qwen3.8-27B, gemma-4-31B, gemma-4-12B, and Mistral-7B, including the cells whose behavior changed in the first study.
- Answered in rounds 1, 2, and 2c for Qwen L31 and L32, gemma-4-31B L32, gemma-4-12B L24, and Mistral-7B L16: additive steering preserves the immediate gap and changes the final-norm gap by −25% to +15%, within the range produced by matched-norm random vectors.
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
