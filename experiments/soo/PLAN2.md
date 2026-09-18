# SOO, second study: measure overlap before explaining behavior

Started 2026-09-18. Status: overlap measurement implemented; model experiments are still pending.

## Writing conventions

Keep this plan and [FINDINGS2.md](FINDINGS2.md) simple, accurate, and understandable.

## Question

Does an intervention make self- and other-referencing representations more alike on unseen prompts? If it does, does that change deceptive behavior without simply damaging the model?

**More overlap means a smaller self/other gap.** We will use “gap” for the measured activation distance and “overlap” for the proposed interpretation. A smaller gap on our probes is not automatically a change in the model's general self-concept.

The first study established useful behavioral effects. It did not establish that those effects were caused by increased self/other overlap. See [AUDIT.md](AUDIT.md) and the audit corrections in [FINDINGS.md](FINDINGS.md).

## What changes from the first study

A fixed steering vector translates both representations by the same amount. At the intervention site, for fixed inputs and the same offset:

```text
(self − offset) − (other − offset) = self − other
```

That operation cannot directly close the gap there. Later layers can respond differently to the shifted inputs, so their gaps still need to be measured. Generated continuations also change the inputs and must be analyzed separately. We will keep additive steering as a comparison, not call it a direct overlap intervention.

Nor does a nonsignificant comparison to one random vector prove that a vector carries no useful information. The Qwen L31 roleplaying comparison did not establish superiority to that control. It does not settle every model, task, or intervention.

## First, fill the single-direction projection gap

The existing `project` hook subtracts the part of an activation pointing along a unit vector `u`:

```text
new_activation = activation − strength × u × dot(u, activation)
```

At strength 1, this removes that direction. Applied to both members of a pair, it removes the same component of their difference. Distance cannot increase at that site, and decreases if the difference has a component along `u`. This is a direct overlap manipulation with a clear mathematical limit: it removes one axis, not every self/other distinction. Shrinking the attention output's gap does not guarantee that the gap shrinks after adding it to the residual stream.

The available result summaries contain projection runs for Mistral, Gemma-2, OLMo, and Muse only. There is no saved projection result for either Gemma-4 size, Llama-2-70B, Qwen2.5-72B, Kimi-Dev-72B, or Qwen3.8-27B. Test this simpler operation before building a more complicated one. Earlier behavioral nulls do not settle these missing cells or tell us how much held-out gap was removed.

## Three alternatives

These are the proposed alternatives, made into testable comparisons. All fitted quantities come from the fit split. Each method must earn its interpretation through the held-out measurements below.

### 1. Remove a self/other subspace

**Idea.** Self/other differences may point along several axes. Opposite differences can cancel in the mean even when individual pairs are far apart.

Collect each pair's activation difference. Use singular value decomposition (SVD) to find the few directions that account for the most squared difference. Keep the differences **uncentered** so the mean difference is not discarded. Ordinary centered PCA describes variation around the mean; it can miss the mean separation we also want to test.

Let `U` contain the chosen perpendicular unit directions:

```text
new_activation = activation − strength × U × Uᵀ × activation
```

At strength 1 this projects out the subspace. Start with 1, 2, 4, and 8 directions, subject to available fit-data rank, and strengths 0, 0.5, and 1. Keep the existing mean-direction projection as a separate baseline: it need not equal the first SVD direction. Do not expand the grid unless the first pass gives a reason.

**Controls.** Random subspaces of the same rank, and a subspace fitted to matched nonsocial pairs. Report the actual size of the activation change as well as rank. A random subspace may disturb the model less despite having the same dimension.

**Test.** Does this remove more held-out self/other difference than one mean direction, with a behavioral benefit beyond the controls? Does it preserve distinctions between unrelated prompts? Removing more dimensions is not a success if the model simply loses useful information.

### 2. Shift only when a probe detects self-reference

**Idea.** A conditional shift can move the self population without moving the other population equally.

Fit a simple linear classifier on base-model activations from the fit pairs: self versus other. For the first implementation, score each unmodified prompt at the chosen last-token site in a preliminary forward pass. Freeze that score for the subsequent intervened pass and generation. This avoids training a last-token probe and silently applying it to every other token position, or feeding altered activations back into its own decision.

Let `g` be the classifier's self probability and `v` the fitted mean self-minus-other vector:

```text
new_activation = activation − strength × g × v
```

With a perfect gate and strength 1, the fitted self mean moves to the fitted other mean. Individual pairs need not coincide, and a gate learned on simple fragments may fail on real prompts containing both self and other. Start with strengths 0, 0.5, and 1. Use the same declared token-position mask as the comparisons.

**Controls.** Always-on shifting, the same gate with a matched-norm random vector, and a gate trained on shuffled labels. A gate given the true self/other label may be used to diagnose the cost of gate errors on the probe set, never as the actual behavioral intervention or a deployable result.

**Test.** Report held-out gate accuracy and scores for both populations, then the gaps and behavior. At behavioral inference, the gate gets only the current prompt. It must not receive the paired alternative, the correct answer, or a deception label. This is probe-gated activation shifting; the older proposal to gate an entire adapter is a different intervention.

### 3. Match self and other means and covariances

**Idea.** Matching the means leaves differences in spread and correlations. Test whether addressing those adds anything.

Start with a small affine map, not a new neural network. Use a fixed low-dimensional basis fitted on training activations; begin with the rank selected in method 1. Within that basis, estimate each population's mean and covariance. Covariance describes how activation coordinates vary together.

Build a regularized whitening-and-coloring map: subtract the self mean, adjust its spread toward the other population's spread, then add the other mean. Regularization prevents tiny estimated variances from producing huge changes. Leave directions outside the fitted basis unchanged.

Apply this map through the **same self-reference gate as method 2**, interpolating between the original and mapped activation at strengths 0, 0.5, and 1. Applying a self-to-other map indiscriminately would also move the other reference population. The learned gate must be used in the real evaluation; oracle labels are diagnostic only.

**Controls.** A mean-only map in the same basis with the same gate, an identity map, and a map fitted after shuffling self/other labels. Keep basis, rank, and position mask matched so the covariance comparison changes one main ingredient.

**Test.** Measure held-out mean and covariance differences in the frozen basis, alongside the paired gap and behavioral/capability measures. Check the map's amplification and activation norms before running generation. Matching two moments is not matching an arbitrary full distribution, and matching populations does not guarantee that corresponding pairs become closer. Keep that distinction in the conclusion. Attempt more elaborate distribution matching only if this simple version yields an interpretable benefit.

## Measure the gap on held-out pairs

### Keep fitting, selection, and final evaluation separate

Use three explicit lists of pair IDs:

| Split | Use |
|---|---|
| Fit | Train adapters; estimate vectors, subspaces, or a conditional intervention |
| Development | Choose layer, strength, and token positions; debug measurements |
| Final test | Evaluate the frozen intervention once; do not tune from its results |

Split by situation and prompt wording, not just by item names or speaker roles. Include both simple self/other fragments and fuller decision contexts. Match each pair on facts, goals, and wording except the intended referent change. Check that the two prompts still make sense.

Existing burglar probes, agentic probes, and Apollo tasks are useful development and regression sets. They are not fresh confirmation data after repeated inspection. Create new situation families and wording for the final test. Record any unavoidable thematic overlap.

For the initial comparison, fit methods separately on the original pairs and the agentic pairs rather than silently mixing them. Compare methods using the same fit set. A data-set comparison is a separate experiment.

### Run matched forward passes

Use the same frozen pairs under the base model and each intervention. Turn dropout off. Keep model revision, precision, chat template, and input text identical across conditions.

For the main measurement, capture the last non-padding prompt token for each member of a pair. State what that token is after applying the chat template; it may be an assistant-header token rather than the last word of the user's prompt. Do not include padding in the loss or align different-length prompts by raw token index.

Capture:

1. The intervention module's output immediately before and after the intervention.
2. The residual stream at the end of that block and later blocks.
3. The final decoder representation, using the same endpoint in every condition.

For LoRA, “before” means the corresponding base-model site. Use the same sites to compare the adapter against the base. Register capture hooks in a defined order so a purported after-intervention measurement actually sees the changed tensor.

Run the intervention with the same layer, strength, and position mask used in the behavioral evaluation. A last-token-only measurement does not mean a last-token-only intervention. Report full-position and prompt-only interventions as separate conditions.

### Report distance and possible collapse

The primary measure is the average paired squared distance:

```text
gap = mean over pairs and activation dimensions of (self − other)²
```

Report the base gap, intervention gap, and paired change at each measured site. Also report:

- Distance between the two population means. This can fall while individual pairs remain far apart.
- Activation norms and variation across different prompts. Shrinking everything or making every prompt alike can reduce the gap without the intended effect.
- Gaps for matched nonsocial control pairs, such as two names or two paraphrases without a self/other change. These help detect general loss of distinctions.

Use float32 for the measurement arithmetic. Compare scales within a model and site; do not rank models by raw MSE. If reporting a normalized gap, state its denominator and show the raw numbers too. A larger activation norm must not be mistaken for a smaller self/other gap.

A reduction imposed by the intervention formula at its own site is a manipulation check. The more informative questions are whether it holds on unseen contexts, survives downstream, preserves other distinctions, and predicts a behavioral difference from controls.

Start with fixed prompts and no generated answers. If a later analysis measures answer tokens, use a fixed shared continuation for the controlled comparison. Separately generated answers differ in content and length; comparing them cannot isolate a representation change on identical inputs.

## Behavior and controls

Begin with one familiar model, Qwen3.8-27B. Use L31 as the first inference-intervention site because its existing additive results provide comparisons; include L32 when measuring existing adapters trained there. Start with full-position interventions and the final prompt-token measurement. Test prompt-only masking separately if a candidate shows an effect or a capability cost. Measure overlap before launching another large model sweep.

For every candidate, compare with the base model, an inactive intervention, and appropriate random/control interventions at the same site and token positions. Use at least three random seeds. Match projection controls on rank; also report actual activation displacement, because equal vector norm or rank need not produce equal disturbance. Include a nonsocial contrast learned from comparable pairs.

For training comparisons, match optimizer steps and batch size explicitly. Equal epochs with different dataset sizes are not equal training budgets. Record dropout and assess held-out gaps with dropout disabled.

Behavioral read-outs:

- Room tasks in both orientations, with honest, deceptive, refusal, and other outcomes separate.
- Roleplaying and insider reporting as development/regression tasks, with independent judge validation. Insider reporting is one shared scaffold with varied supplied histories.
- A fresh family of behavioral situations for any new transfer claim.
- Ordinary generation and coding alongside multiple-choice capability checks. Report completion, truncation, and refusal rates.

Use the repaired sandbagging analysis if that task is revisited. Missing private or final answers remain missing evidence. Do not use the existing narrow MMLU slice as a general capability score. The coding ports remain tests of their specific bounded harnesses.

## How to interpret the result

| Held-out gap | Behavior | Interpretation |
|---|---|---|
| Smaller | Less deception, controls weaker, capabilities retained | Evidence consistent with overlap contributing; other mechanisms still need to be excluded |
| Smaller | No clear behavioral change | This amount and location of gap reduction is insufficient on these tasks |
| No clear reduction | Less deception | The effect is not explained by the overlap measured here; other sites or contexts remain possible |
| Smaller | General distinctions or capability collapse | Gap reduction through damage is not a successful honesty intervention |
| Either | Random/control intervention behaves similarly | The proposed self/other-specific explanation is not established |

A dose-response relationship across gap and behavior would strengthen the evidence. It would not alone prove mediation. Start with the controlled comparison; reserve more elaborate causal tests for a result that survives it.

## Work order

1. **Implemented:** add held-out overlap measurement to the existing intervention setup. Check inactive hooks, padding, capture order, and the fixed-translation identity on small examples. Check that full projection removes its selected component and that an inactive gate leaves activations unchanged. See [OVERLAP.md](OVERLAP.md) for the runner, input format, and software checks. The gate check uses an artificial gate; a trained probe remains future work.
2. Measure the base, existing additive steering, and existing original/agentic adapters on development pairs. This asks what the old interventions actually changed. Use the new runner described in [OVERLAP.md](OVERLAP.md); the old `latent_soo.py` full-tensor MSE is not this measurement.
3. Fill the single-direction projection gap, then test subspace removal, gated shifting, and covariance matching in that order. Use the same development pairs and behavioral examples. Report every tested cell. Stop a branch if its measured benefit is only collapse or cannot be separated from its controls; do not build a more elaborate version merely to rescue the hypothesis.
4. Choose a candidate using a rule written before viewing its final test results. Freeze its settings, primary site, behavioral endpoint, meaningful-effect thresholds, capability margin, and comparisons in a dated entry. This document is a study design, not a claim that those numerical choices have already been preregistered.
5. Run the untouched final test, repeat trained candidates across at least three seeds, and report successes and failures together. A vector fitted deterministically to fixed data has no training-seed uncertainty; assess its fit-data sensitivity by resampling situation families. Expand to a second model, initially Gemma-4-31B, after the first study is interpretable.

Follow [PROTOCOL.md](PROTOCOL.md) for paired analysis, judge checks, and provenance. Here, “not separated from random” means specificity is unestablished, not disproved. Previously inspected examples cannot become untouched confirmation data by being assigned a new index range.

For continuous gaps, use paired intervals over the same pairs; account for shared situation families. For binary behavior, use paired comparisons. Do not pool repeated seeds or orientations as independent examples. Identify the small confirmatory comparison family in advance and adjust for multiple testing. Report the random-control distribution, not just a favorable seed.

Save per-pair measurements, raw completions, completion status, split IDs, model/code revisions, dataset and intervention hashes, chat settings, token budget, seeds, and all selected settings. Keep second-study outputs under a separate run namespace so old records cannot be overwritten or silently resumed.
