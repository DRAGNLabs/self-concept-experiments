# Fixed half-strength transfer — 2026-10-27 Mountain

## Authorization and question

The user requested proceeding with the half-strength result on larger evaluation
sets. The prior fixed-grid development sweep found alpha 0.5 identical to the
adapter on all 432 payoff responses/tokens, including 108 private and 216 factual
answers. It did not match the adapter on the small historical main-task check:
offset 0.5 remained deceptive on all 16, adapter honest on all 16.

Freeze alpha **0.5**, with no further selection or tuning. Run exactly two new
conditions: the original nine-instance validation prompts (1,296) and six-profile
numerical-preference prompts (2,592). Maximum **3,888 new scored generations**,
plus existing fixed-batch/historical controls. Reuse saved baseline, full adapter
and alpha-1 offset for each suite. The name/column diagnostic, alpha 0.75, new
doses, new prompts, training and mechanistic experiments are outside this batch.

The dose was chosen using development outputs. Other conditions on both target
sets have already been inspected; this is a fixed-dose transfer check on existing
evaluation sets, not a new untouched task holdout or a general equivalence test.
The original combined gate remains failed. Recommendations are action choices,
not assertions of truth; no honesty claim follows from recipient-optimal choices.

## Frozen execution

Use the exact immutable datasets, native prompt construction, keys, suite order,
batch membership and inference settings of their existing runs. No new generator,
tokenizer, model runner or scoring changes. Gemma pinned revision, bf16 SDPA,
thinking disabled, greedy generation, batch 8, cap 32, same offset artifact and
ten v_proj sites/positions; only the already-supported alpha is set to 0.5.

Both execution snapshots and analysis/controller/protocol are frozen before either
submission. Reuse the passing original validation private/factual gate and all
120 numerical baseline checks. Require the previous dose endpoint/repeat checks
to have passed. Verify current inference-source hashes exactly match the selected
half-dose development run; native tokenization is unchanged and existing inputs
provide the exact cross-run comparison. No new task-feasibility threshold is added.

## Analysis and safeguards

Keep every trial and error. For baseline, adapter, half offset and full offset,
report private accuracy, both factual accuracies, opposed recipient-optimal
recommendations, aligned recommendation errors, invalid/dominated answers, and
instance/role/order strata. For numerical data also report profile and competing
rule matches. Primary descriptive contrasts are half-minus-adapter and half-minus-
full-offset private accuracy, paired losses/gains, and recommendation differences.
Report answer/token disagreements directly, not only equal aggregate rates.
Keep private/factual correctness attached to each matched recommendation; do not
select or filter contexts by correctness. No prompt-level significance/equivalence
test: repeated rotations and reused templates are not independent tasks.

Independently rescore and audit frozen artifacts, table keys, exact native inputs,
batches, model/settings/package versions, and actual offset artifact hashes.
Numerical runs must have exactly the selected dose's inference-source hashes.
For older validation comparisons, permitted source differences are only the
previously checked alpha argument and the two inference-unused diagnostic modules;
the prior endpoint bridges already checked that transition. Require all new
fixed-batch sentinels and all 32 historical regression records to reproduce the
selected half-dose development run exactly, including generated tokens. The old
generic auditor's full-strength historical-label expectation is retained as
descriptive metadata; it is not the acceptance rule for a half dose.

Any execution, hash, input or repeat discrepancy is recorded as inspection-needed,
not silently waived, filtered or rerun. Both suites are analyzed independently
after the two jobs complete; a behavioral loss in one does not remove the other.
No automatic follow-up jobs. If private/factual errors emerge or recommendations
move less, report that limitation. If the development pattern transfers, narrow
the earlier full-strength comparison accordingly without claiming shared internal
mechanisms or transfer to the historical deception task.
