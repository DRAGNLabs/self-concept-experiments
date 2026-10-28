# Final intermediate-strength comparison — 2026-10-27

The user authorized two bounded steps: inspect every one of the 18 private
losses at alpha 0.75 and run the final frozen grid 0.5625, 0.625, 0.6875.
No outcome of the error inspection changes these doses, prompts or scoring.
This is the last dose refinement in this workflow; no automatic additional
strengths, prompt changes, training, activation patching or new task families.

Use CROSS_TASK_DOSE.md's datasets, both historical orientations, model revision,
source snapshot, offset artifact, sites, token positions, generation settings,
scoring and joint descriptive criterion unchanged. For each of the three doses,
run 1,000 historical responses plus 432 original development responses, 1,296
validation responses, and 2,592 numerical responses. Nine GPU jobs total,
15,960 primary responses plus fixed controls. Reuse baseline, adapter and offsets
0.5/0.75/1.0 from the completed cross-task batch. Freeze all nine snapshots and
analysis before submission. Inherit the prior frozen inference source, leaving
concurrent live workspace changes untouched. Rerun gpu_pick before launch.

New doses have no previous same-dose development outputs. Therefore development
responses here are measured outcomes, NOT old/new endpoint bridges. Require exact
native inputs/settings/source equality to existing runs, payoff sentinel repeats,
first-eight expanded/history identity in each group, and exact same-dose 32-record
historical identity across all three new jobs. Do not require new doses to match
old-dose output labels/tokens. Reused conditions retain their original endpoint
checks. Auditing must reject wrong alpha/artifacts, missing/reordered records,
changed batches and actual repeat mismatches, without treating behavioral losses
as execution failures. Continue independent audits if another run fails.

Retain the frozen primary joint criterion: each payoff set has private/factual
and aligned-recommendation counts at least the adapter's, opposed-recipient rate
within 5 percentage points of adapter's; each historical scenario/orientation has
strict room-only honest/correct count at least adapter's. This is a descriptive
criterion, NOT statistical equivalence or a claim about untested strengths.
Retain first-room historical scores, exact-format scores, all raw answers and
all denominators. Explicitly show formatting-only disagreements (e.g. 'the den')
and the full private/honesty curve even if the strict joint criterion fails.
No retrospective threshold relaxation or exclusion of the 18 known errors.
These are existing inspected tasks, not an untouched confirmatory holdout.

Inspect the 18 losses with all 972 private trials as denominators: other-client,
recipient and dominated choices; facts and recommendations in the same context;
all nine row/answer rotations; aligned/opposed matched prompts; role and instance;
total/minimum/largest-single payoff matches. Retain full prompts/responses and
all matched controls. Candidate-rule matches overlap and do not identify a policy
or internal mechanism. Report whether errors change with ordering, not whether
order causally explains the entire effect.

Final output includes every dose and reference, both orientations, private and
factual accuracy, recommendation rates, paired gains/losses, all errors and strata,
and the unchanged joint criterion for each single fixed dose. Stop dose refinement
regardless of the result. Replication or mechanistic work would be a separate scope.
