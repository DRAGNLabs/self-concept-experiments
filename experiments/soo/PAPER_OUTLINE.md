# Draft paper outline: self–other overlap

Prepared October 23, 2026. This is a proposed paper structure, not a report of new experiments. Quantities below come from the existing records; the full evaluation matrix has not been independently recomputed for this outline.

## Recommended framing

**Working title: What Do Self–Other Overlap Adapters Change? Representation Collapse and Distributed Behavioral Effects**

Alternative: **Disentangling Representation Collapse and Behavioral Change in Self–Other Overlap Fine-Tuning**.

The central question is whether the behavioral changes produced by SOO training are explained by the intended selective convergence of self/other representations. The strongest contribution is a controlled empirical investigation: measure the representation change, test a simple proposed explanation, localize the effective parameter updates, and test whether fixed offsets can reproduce their behavioral effect.

Proposed thesis:

> In our last-token implementation of SOO, large reductions in self/other distance often accompany a broader loss of distinctions at the training site. Across five selected adapters, earlier-layer updates match the reduction in burglar-task deception, whereas training-layer-only updates do not. Fixed offsets extracted from trained updates reproduce the behavioral criterion in three models, with important exceptions. These findings constrain the mechanism of the tested adapters without establishing selective self/other overlap as the mediator.

This supports a mechanistic case study with positive empirical contributions. A complete explanation of the downstream computation remains open. “SOO does not work,” “overlap is unnecessary,” and “we have identified the honesty mechanism” would all exceed the evidence.

Use the five-model mechanism study as the main paper. Include the initial replication as motivation and validation, and give agentic transfer a compact boundary section. The full ten-model steering search, coding experiments, and chronological research narrative belong in appendices or a companion report. The older PLAN.md proposals about a generalization certificate or an improved agentic method have been superseded by later results.

## Provisional abstract

Self–Other Overlap (SOO) fine-tuning seeks to reduce deception by minimizing distances between activations elicited by self- and other-referencing prompts. We investigate what this objective changes in a last-token implementation. Across four measured models, reduced self/other distance accompanies substantial loss of prompt-dependent variation at the training site; name-control measurements show that this contraction is not consistently selective. Imposing constant attention outputs on three base models does not reproduce their adapters’ behavioral effects. Module-subset experiments across five selected adapters instead locate updates sufficient to match burglar-task deception reductions below the training layer. Compact bands reproduce the behavioral criterion in four models, and fixed offsets extracted from trained updates do so in three. Gemma-4-12B exhibits a preservation tradeoff under fixed offsets, while Kimi-Dev-72B requires more distributed query and value updates among the subsets tested. These exploratory results distinguish objective minimization, measured representation change, and behavioral improvement, and motivate stronger controls for interpreting representation-based safety interventions.

## Main paper structure

An illustrative eight-page body follows; this is a writing budget, not a claim about any particular venue’s format. References and appendices are additional.

### 1. Introduction — approximately 0.75 page

**Question:** What must be demonstrated before a representation-based intervention can be interpreted as changing the proposed mechanism?

- Introduce SOO’s motivation and the reported reduction in deceptive answers. Attribute the hypothesis to [Carauleanu et al.](https://arxiv.org/abs/2412.16325); keep the neuroscience analogy brief.
- Distinguish three propositions: the objective decreases; representations change selectively in the intended way; that change explains the behavior. The first does not establish the other two.
- Give the motivating observation: on Mistral, a lower learning rate reaches very small SOO loss without the desired behavioral change. This motivates measurement and intervention, not a conclusion about optimization drift.
- State the scope immediately: a last-token SOO implementation, selected adapters, controlled toy scenarios, and exploratory follow-ups.
- End with three contributions: a measurement of nonspecific contraction; intervention-based localization of effective updates; and a fixed-offset substitution experiment with cross-model exceptions.

**Suggested opening sentence:** “A training objective can produce a desired behavioral change without identifying the representation that mediates it.”

### 2. Background and experimental design — approximately 1.25 pages

**2.1 Objective and implementation.** Define paired prompts and the activation site. For a pair \(x_i^s,x_i^o\), let \(a_{L,t}(x)\) be the chosen attention-output vector. Write the last-token objective as

\[
\mathcal L_{\mathrm{SOO}}
=\frac{1}{Nd}\sum_i\left\|a_{L,t_i^s}(x_i^s)-a_{L,t_i^o}(x_i^o)\right\|_2^2.
\]

Each endpoint is the corresponding prompt’s last valid token after chat templating. It can be an assistant-header token. Report the actual endpoint tokens. Explain that the loss is read at one layer while LoRA updates earlier query/value projections through backpropagation. The absence of trained updates above that layer follows from the computation graph and initialization; it is a sanity check, not the principal discovery.

Explicitly distinguish the initial full-sequence implementation from the last-token objective used for the mechanism study. The former computes unmasked MSE on right-padded tensors; its alignment was an implementation choice, not a verified specification of the original work. A last-token comparison changes both alignment and supervision positions. Do not attribute differences solely to padding without a masked, aligned control.

**2.2 Models and selected adapters.** One table should give exact checkpoint/revision, total layers, loss layer, LoRA targets/rank/scaling/dropout, optimizer settings, training pairs, seed coverage, and selection history. Core cases: Mistral-7B L16, Gemma-4-12B L24, Gemma-4-31B L32, Qwen3.8-27B L32, and Kimi-Dev-72B L56. Explain fused query/key/value channels on Qwen. Distinguish the count of installed modules from modules with nonzero trained updates.

Do not imply five seeds everywhere. The recovered Mistral L16 comparison covers five seeds and 250 examples per condition; Gemma-2 L14’s early comparison covers two seeds and 50 examples. Localization and offset experiments use one selected adapter seed per model. Kimi’s layer was selected from nine single-seed candidates.

**2.3 Behavioral measurements.** Describe the burglar scenario, paired room-order reversal, Treasure Hunt, and Perspectives. Report true-room answers, decoy-room answers, refusals, other outputs, and truncation separately over all examples. Use “deceptive answer” as the task’s operational label, not a verified claim about intent. Explain Mistral’s intent-language check and Kimi’s final-answer parsing. Perspectives is a narrow perspective-tracking control, not a general theory-of-mind or capability certificate.

**2.4 Representation measurements and controls.** Use fixed identical prompts across conditions; dropout disabled; 32 self/other development pairs and 32 name-control pairs across 16 situation families. Measure paired MSE, activation norms, and across-prompt variation at the attention output, downstream residual sites, and final normalization. Inspect content-token endpoints as well as the final template token. Name substitutions test specificity to this contrast, not preservation of all unrelated information.

**2.5 Analysis.** Separate exploratory selection from untouched evaluation. Use paired outcome comparisons and family-aware intervals where appropriate; never count orientations or seeds as independent new examples. Report effect sizes and continuous outcomes alongside the exploratory reproduction thresholds. Launch-time plans are internal prospective records, not public preregistration. Inspect per-round rules: the ten-point behavioral tolerance does not constitute a statistical equivalence test.

Sources: [training code](../../src/selfconcept/soo/train.py), [PLAN.md](PLAN.md), [PLAN2.md](PLAN2.md), [measurement specification](OVERLAP.md), [protocol](PROTOCOL.md), and the current published implementation notes.

### 3. Behavioral improvements are real but evaluation-sensitive — approximately 0.75 page

**Purpose:** Establish that there is a substantive behavior to explain while showing why deception rate alone is insufficient.

- Lead with the stable Mistral L16 result: original-order deception falls from 90.4% to 9.9% ± 6.1 across five seeds. Report mirrored results and capability changes from their actual recovered records.
- Contrast this with the initial Mistral L19 full-sequence runs: low deception coincides with 58–90% “other” responses. The number alone would have hidden damage.
- Give one positional-confound example: OLMo’s baseline main-task deception changes from 14% to 94% when room order reverses. Avoid presenting original-order truthfulness as a general trait.
- Introduce the five selected adapters that will be dissected. Their successful selection is part of the study design, not an estimate of the success rate of arbitrary SOO training.

**Figure 1:** Stacked behavioral outcomes for baseline, a damaged condition, a successful adapter, and mirrored counterparts. Include a short anonymized example of a true-room answer with contradictory deceptive language.

Keep full layer sweeps and quantization/epoch ablations in the appendix. Describe failures under the tested implementations; do not call them exact reproductions of unspecified original implementation details.

Source: [first-study findings](FINDINGS.md), especially the setup, Mistral, audit, and capability sections, plus the recovered first-study archive linked from the published post.

### 4. Minimizing the objective produces broad contraction — approximately 1 page

**Question:** Is the measured self/other convergence selective?

- Explain the deterministic constant solution: if every tested prompt yields the same vector at the measured endpoint, every paired distance is zero. This is a mathematical property of the objective; observing near-collapse is the empirical result.
- Present Qwen’s nine adapters first: roughly 99.98% reduction in the self/other gap, 99.93% in the name-control gap, and 99.9% in across-prompt variation at the training site.
- Show that contraction persists at the last user-content token and extends to other measured models.
- Preserve the Gemma-4-12B exception: name-control changes differ markedly across seeds, including an increase from a very small baseline. Report raw gaps and individual seeds; the positive mean percentage change must not be hidden behind “all gaps collapse.”
- Show at least one downstream comparison. Local collapse does not mean the entire residual stream or all downstream information becomes constant; downstream self/other and name-control changes can diverge.

**Figure 2:** Self/other gap, name-control gap, and prompt variation by site and condition, with seed points and family-based intervals. Show raw scales or a companion table so near-zero denominators do not create a misleading percentage narrative.

**Licensed conclusion:** Near-zero loss and local gap reduction do not establish selective self/other convergence. Kimi has no corresponding representation measurements in the present mechanism study.

Sources: [FINDINGS2.md](FINDINGS2.md), rounds 2–2c; their per-run analysis files under [study2 results](results/study2/).

### 5. Which updates reproduce the behavioral effect? — approximately 1.5 pages

**5.1 Constant replacement.** Replace the base model’s attention output at the training layer with the adapter mean, base mean, zero, or matched-norm random constants. Distinguish replacement from addition and identify the intervened positions.

On the two Gemma models, response-position replacements leave the main task fully deceptive. Mistral’s apparent improvement is largely confused true-room answers with deceptive language. Report the small Gemma-4-12B movement under all-position replacement separately.

**Inference:** The tested replacements do not reproduce the adapter. This is a restricted sufficiency test; it does not prove activation collapse is unnecessary, since imposing a constant on the base model does not recreate all interactions in an adapted model.

**5.2 Module subsets.** Retain only training-layer, below-training-layer, query, or value updates. Across all five selected adapters, earlier-layer updates match the main-task deception reduction and training-layer-only updates do not. State the Gemma-4-12B exception to the full reproduction criterion: its below-training-layer subset has 78.4% mirrored Perspectives accuracy.

**5.3 Contiguous bands.** Present the smallest successful subsets found within the search, not globally minimal circuits:

| Model | Successful subset among those tested | Boundary to report |
|---|---|---|
| Gemma-4-12B | Four value-projection modules in layers 19–23 | 0% deception and 100% Perspectives in both orders; improves these outcomes over the full adapter |
| Mistral-7B | Nine value-projection modules, layers 5–13 | Main-task criterion reproduced; Treasure Hunt and intent-language outcomes differ from the full adapter |
| Gemma-4-31B | Ten value-projection modules in layers 20–31 | Main-task effect reproduced; the subset leaves Treasure Hunt deceptive |
| Qwen3.8-27B | All adapter modules in layers 24–31 | Main-task effect reproduced; fused value-only channels explain only part of the broader localization result |
| Kimi-Dev-72B | Query and value updates across layers 0–55 | No narrow tested subset reproduces the full effect; query-only and value-only conditions increase deception |

**Figure 3:** A layer/module map alongside behavioral outcomes for base, full adapter, training-layer-only, below-layer, and selected-band conditions. Mark absent/untrained modules separately from measured null effects.

Keep the causal language precise: retaining a subset tests sufficiency within this intervention setting. Necessity requires the relevant removal/complement comparison; exhaustive minimality is not established. Continued collapse under below-layer subsets prevents the claim that the representation collapse has been causally eliminated.

Sources: [constant round](CONSTANT_ROUND.md), [layer round](LAYER_ROUND.md), [bisection](BISECT_ROUND.md), [second bisection analysis](results/study2/bisect2-gemma4-31b-L32-20260928T230232Z/output/analysis.md), [Qwen localization](results/study2/qwenloc-qwen38-27b-L32-20260929T180213Z/output/analysis.md), [Kimi localization](results/study2/kimiloc-kimi-dev-72b-L56-20260930T223131Z/output/analysis.md).

### 6. Fixed offsets reproduce some trained effects — approximately 1 page

**Question:** How much input dependence is needed to reproduce the observed behavior?

Define a trained projection update as \(\Delta_l(h)=s_lB_lA_lh\). Average its realized outputs over the stated extraction prompts and positions to obtain \(\bar\Delta_l\). Add \(\alpha\bar\Delta_l\) at the corresponding base-model projection outputs without loading the adapter. Describe the extraction forward-pass configuration and token averaging exactly.

- Contrast these adapter-derived offsets with the first study’s self-minus-other activation vectors. They are different interventions with different provenance and sites.
- Fixed offsets meet the main behavioral reproduction criterion on Mistral, Gemma-4-31B, and Qwen. Report the reference used for each comparison: a band and a full adapter are not interchangeable.
- Prompt-position interventions carry effects in the tested cells; report the actual model-specific masks and doses. Do not turn this into a universal claim about how SOO processes context.
- Gemma-4-12B is partial: zero main-task deception accompanies refusals and about 87% mirrored Perspectives under the all-position offset. Some stronger random offsets also substantially change its behavior; one tested random dose meets the reproduction rule. Direction specificity therefore cannot be claimed from the deception endpoint alone.
- Kimi’s tested fixed offsets do not reproduce the full adapter while satisfying the preservation conditions. Content-only offsets give partial changes. This limits the fixed-offset account; it does not prove that every possible constant approximation must fail.
- Gap contraction under the tested offsets is not clearly selective relative to the name controls. Equal behavioral performance does not show that an offset and an adapter implement the same computation.

**Figure 4, or combine with Figure 3 for space:** Behavior and Perspectives versus dose, with real and matched random offsets and prompt/response masks. Show failures and partial outcomes, not only the selected successes.

Sources: [band-steering design](STEER_ROUND.md), [31B dose/position results](results/study2/dose-gemma4-31b-L32-20260929T221420Z/output/analysis.md), [12B dose/position results](results/study2/dose-gemma4-12b-L24-20260929T221158Z/output/analysis.md), [Mistral dose/position results](results/study2/dose-mistral-7b-L16-20260930T014849Z/output/analysis.md), [Qwen offsets](results/study2/steer-qwen38-27b-L32-20260929T221532Z/output/analysis.md), [Kimi joint offsets](results/study2/kimijoint-kimi-dev-72b-L56-20261001T202619Z/output/analysis.md).

### 7. Transfer, interpretation, and limitations — approximately 1.75 pages

**7.1 Transfer is task-dependent.** Use one compact table from the first study to show why burglar success is not a general honesty score. On Qwen, original-pair training improves insider-report disclosure but leaves judged roleplaying deception near baseline; agentic-pair training improves roleplaying while increasing concealment in the insider scaffold. All three pair sets achieve the toy-task result.

Useful descriptive anchors: roleplaying deception is 48.8% at base, 51.2% with the original adapter, 27.0% ± 3.1 with agentic pairs, and 32.3% ± 4.4 with mixed pairs. Insider concealment is 54.3% at base and 24.3% with the original adapter, versus 64.2–89.0% across the six agentic/mixed seeds. Keep explicit, implied, concealed, and invalid categories visible.

Report the 246 role-disjoint examples separately; role disjointness is not scenario-family disjointness. The 173 insider continuations share one scaffold and supplied trading histories. These runs evaluate reporting, not autonomous decisions to make the trade. Unequal optimization budgets and unfinished judge validation limit causal attribution to pair content. Do not claim that visible reasoning length identifies the internal cause.

**7.2 Interpretation.** The positive finding is that the behavioral effects can be localized and, in several cases, reproduced with a simpler inference intervention. The representation measurements identify a weakness in interpreting an unconstrained distance objective. They do not settle whether self/other representations play some role elsewhere in the network.

**7.3 Limitations.** Include selected adapters; adaptive layer/dose searches; one seed per model for localization; sparse random-offset draws; shared pronoun/name constructions; limited construct validity of room labels and intent heuristics; missing Kimi gap measurements; missing Qwen/Kimi constant replacements; and no general capability evaluation of the second-study bands and offsets. Passing Perspectives is insufficient to call an intervention capability-preserving.

The October random-direction pilot found little change on sampled generated ARC/GSM8K questions, but it tested a different, first-study Gemma intervention. It cannot certify the second-study offsets, and the earlier insider outcome was not rerun in the same runtime. Include only if relevant to the transfer discussion.

**Closing claim:** These experiments provide a way to distinguish successful optimization, representational selectivity, and behavioral efficacy when evaluating safety interventions. Selective overlap as a causal mediator remains an open empirical question.

Sources: [FINDINGS.md](FINDINGS.md), final round-8 tables and audit corrections; [AUDIT.md](AUDIT.md); [random-direction capability pilot](RANDOM_CAPS_PILOT.md).

## Appendix plan

- A. Exact prompts, token endpoints, model revisions, training configurations, and differences from the original SOO implementation.
- B. Complete replication and steering search matrices, with actual sample/seed coverage and original/mirrored conditions.
- C. Full localization, complement, bisection, offset-dose, and token-mask results; exact verdict rules and threshold sensitivity.
- D. Raw and normalized gap measurements by seed/site; name controls; subspace-removal pilot. A fixed addition preserves the immediate paired difference algebraically, while downstream transformations can change it.
- E. Agentic transfer, role-disjoint results, judge validation, paired estimates, and training-budget controls where available.
- F. Coding and capability boundaries. Use repaired EvilGenie results only for quantitative claims; distinguish failed hacking attempts, passing cheats, and ordinary mistakes. Include sandbagging only with repaired parsing and explicit missing-answer denominators.
- G. Reproducibility inventory, hashes, artifact availability, and which records are recovered versus absent. A frozen exploratory plan must not be relabeled preregistration after publication.

## Most valuable work before submission

These are proposed additions, not completed results or prerequisites for writing the initial preprint.

1. **Finalize the evidence table and scoring validation.** Resolve the stale summaries against run outputs, validate confusing/invalid/truncated examples with a blinded human sample, and report seed/sample coverage per claim. Recompute paired estimates for the few main comparisons. This has higher immediate value than another broad layer sweep.
2. **Repeat the central mechanism comparisons.** On two contrasting models, repeat full adapter, training-layer-only, below-layer, selected band, and fixed offset across at least three adapter seeds. Add several matched random-offset draws. Freeze the selection rule before evaluating genuinely fresh prompt families; previously inspected items cannot become untouched tests by reindexing them.
3. **Measure the actual proposed substitutes’ costs.** Evaluate ordinary generated answers and a manageable coding/task suite under the selected bands and offsets. Define a meaningful noninferiority margin before claiming preserved capability. First-study full-adapter scores do not transfer automatically to these new interventions.
4. **Add an objective control that can distinguish explanations.** With matched update budgets, compare self/other pairs with name-swap, shuffled-pair, and self/self-with-independent-dropout training. Add a variance/preservation constraint if pursuing a new method. This tests whether the self/other pairing contributes beyond generic invariance pressure. An anti-collapse objective alone is not an established improvement until behavior and preservation are evaluated.
5. **If making a stronger causal claim, intervene on the mediator.** Attempt restoration of prompt-dependent activations while retaining the effective updates, with matched disruption controls. This could test necessity more directly, but restoration can itself disturb computation. Treat it as a new experiment, not an implication of the existing constant replacements.

For a short workshop submission, prioritize items 1–3 on a small model set and narrow the claims. A larger conference version would benefit especially from item 4 and a fresh behavioral family. A successful mechanism study does not require inventing an improved alignment algorithm, but any claim to have improved SOO does require that additional evidence.

## Short workshop version

Use a roughly four-page body if the venue permits: introduction/question (0.5), methods (0.75), nonspecific contraction (0.75), localization and fixed offsets (1.5), limitations/discussion (0.5). Lead with Gemma-4-12B and Gemma-4-31B as a clear contrast, show the five-model summary in one table, and move detailed replication, agentic transfer, and all coding work to the appendix. Keep Kimi as an explicit boundary case rather than presenting the offset result as universal.

## Source and citation notes

The published posts checked for this outline were [Self-Other Overlap Doesn't Quite Work How It's Claimed To](https://www.jansennye.com/research/self-other-overlap-doesnt-quite-work-how-its-claimed-to/) and [What Do Self-Other Overlap Adapters Actually Do?](https://www.jansennye.com/research/what-do-self-other-overlap-adapters-actually-do/). The first includes an October 21 correction to sample/seed coverage; the second qualifies earlier claims that collapse was unnecessary. Those qualifications should supersede stronger wording in old logs and drafts. Later localization and offset results are present in run analyses even where FINDINGS2.md still calls them pending.

Related-work starting points, rather than a completed literature review:

- [Carauleanu et al., Towards Safe and Honest AI Agents with Neural Self-Other Overlap](https://arxiv.org/abs/2412.16325): the intervention and hypothesis being investigated.
- [Panickssery et al., Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681): contrastive activation steering; distinguish that construction from extracting fixed updates from trained adapters.
- [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906): constant-representation collapse and explicit variance preservation. The broad collapse problem is established prior work; the proposed novelty is its empirical manifestation and behavioral dissection in these SOO adapters.

Before claiming novelty, extend the search to representation-based deception interventions, adapter-to-steering conversions, and causal validation of mechanistic interpretations. Cite the exact upstream evaluation datasets and model releases used, rather than relying on the blog summaries for methods.
