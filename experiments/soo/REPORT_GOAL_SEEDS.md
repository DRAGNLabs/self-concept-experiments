# Report objective comparison across training seeds, 2026-10-27

The user authorized continued investigation after the goal-reminder result.
The next bounded step is replication on both other existing adapters, seeds 1
and 2, before further prompt changes. Beneficiary reversal and wording/position
controls remain future work, not automatic jobs in this batch.

## Fixed comparison

Reuse every prompt, native input, batch, sentinel and decoding setting from
`report-goal-scope1-batch-20261027T230343834417Z.json`. Eight variants, six already
evaluated lexical instances, all roles, recipient alignments, table and answer
orders, and all four queries. The inherited development/reserved names identify
two previously evaluated sets; neither is a new holdout.

Use full Gemma-4-31B L32 adapters from checkpoint seeds 1 and 2, with matching
training configurations and distinct weight hashes. Seed 0 must match the
completed parent exactly. Freeze both new adapters before either runs. No new
training, offsets, prompt selection, seed selection or strength tuning.

The completed parent supplies the shared base and seed-0 reference. Reverify its
hashes, output audits and both 952-check competence gates before submission and
final analysis. Base is reused, not rerun or counted as an independent replicate.
No outcome from either new adapter gates inclusion of the other.

Four GPU runs: two seeds × two sets × 3,456 primary responses = **13,824 new
primary responses**. Each run also repeats the original 32 complete sentinel
batches before and after inference (512 responses) and generates the same 32
historical checks. Fresh output only; no silent retries or snapshot edits.

Use the unchanged parent inference source and execution function: pinned model
revision, bf16 SDPA, thinking off, greedy batch eight, cap 32. Consult gpu_pick.py;
retain one A100, 4 CPUs, 96 GB, 60 minutes and dw-2-4 exclusion to match prior
hardware even when another GPU type has a shorter queue.

## Integrity and analysis

All runs are audited for exact native inputs, plans, batches, padding, completion
coverage, source, model/library settings and the intended adapter hash. Sentinel
changes stop final pooled interpretation. New seed historical responses need not
equal seed 0: preserve and report those differences. The 32 historical responses
must match within each seed across its two runs; mismatch stops final pooling,
with raw outcomes and successful audits retained. This is an integrity check,
not a gate favoring honesty or any particular recommendation.

Apply unchanged independent scoring and the full parent analysis separately to
each seed versus the shared base. Keep all variants, role, instance and presentation
strata, aligned recommendations, private/factual errors, invalid/dominated answers,
paired answer transitions and joint communication/control outcomes. Retain seed-0
summary alongside seeds 1 and 2; do not report only a mean or a favorable seed.

Primary replication target: original_reminder minus original opposed-recipient
recommendation rate, raw and baseline-adjusted, separately for each seed and set.
All parent payment/scope contrasts and interactions are retained as secondary
descriptive results. Also compare each new adapter answer with seed 0 on every
input. There is no favorable-effect threshold or statistical significance gate.

The seed-0 comparison was discovered adaptively. Seeds are new evaluations on old
prompts, not a new task-family replication. Three adapters and six lexical cases
do not establish broad generalization or mechanism. A general reminder inside
report context can still inherit report scope. Behavioral recommendations are
not direct honesty measurements. No prompt-level independent-sample p-values,
equivalence claims, or internal-mechanism conclusions.
