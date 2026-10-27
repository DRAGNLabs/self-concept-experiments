# Payoff presentation follow-up 2

## Scope and authorization — 2026-10-27 Mountain

The user asked to keep investigating the completed held-out behavioral difference.
First inspect all saved development/validation outputs, then run this bounded
presentation diagnostic. This is exploratory follow-up after seeing prior results;
it is not new held-out validation or a repair of the failed combined protocol.

The saved-output review finds recipient-optimal opposed recommendations in all
nine validation instances for the adapter (16–18/18 per instance), with 14/18
instance/role groups choosing that answer at every row/answer rotation. Changing
the recipient objective changes 156/162 paired recommendations and zero private
choices. Offset private choices change in 25/162 corresponding pairs; 19/49 private
errors occur even with aligned objectives. Of all 49 private errors, 47 select
the unique total-payoff maximum. This latter association is post hoc and may be
confounded with client identity, names, payoff magnitudes or display position.

## Frozen comparison

Reuse the exact 1,728 development prompts and batch composition of the completed
baseline name/column diagnostic (`payoff_name_column1`). Run only the full adapter
and fixed offset on that same four-cell factorial. Reuse the completed baseline.
The existing dataset's baseline-only label describes the original closed study;
new provenance must explicitly identify `payoff_presentation2`. Default launching
on that dataset remains baseline-only unless this specific follow-up is supplied.

All three development instances, both represented roles, both alignments, three
row rotations, three answer rotations, and all four query types remain. Column
swaps move headers and values together; name swaps rename clients consistently
without changing indexed payoffs/goals. No wording, keys, exemplars, filtering,
token limit, decoding, model, adapter, offset, or inference implementation changes.
Gemma-4-31B-it pinned revision; bf16 SDPA; greedy; thinking disabled; batch 8;
32 generated tokens; offset alpha 1.0 at the same ten v_proj sites.

Maximum new scored generations: 2 × 1,728 = **3,456**, plus the existing fixed-batch
sentinels and 32 historical regression prompts per condition. No additional
conditions, dose sweeps, training or activation patching are released automatically.

## Analysis and checks

Report every answer. Within each of the four column/name cells, report private
accuracy, facts, and opposed recipient-optimal recommendations for base, adapter,
offset. Compare adapter-minus-offset private accuracy and adapter-minus-baseline
recipient-optimal recommendations. Also report instance/role/alignment cells and
paired column/name changes for each intervention. No prompt-level significance or
equivalence claim; only three development instances with repeated presentations.

Require matching prompt tokens and batch membership across conditions. Reconstruct
the factorial and answer keys independently, rescore outputs, verify frozen hashes,
and retain full errors. For each condition, compare the unchanged 432-prompt subset
against that condition's original development run, both answers and generated tokens.
Retain the baseline inactive-hook check and all historical/sentinel checks.
Any discrepancy produces an explicit inspection-needed status, not silent acceptance,
filtering, reruns, or automatic continuation. Diagnostic accuracy is an outcome,
not a new task-acceptance or validation-release gate.

If the contrast changes substantially with names/columns, narrow it to that
presentation dependence. If it persists, that rules out those particular swaps as
a complete explanation, not superficial cues generally or unequal intervention
strength. Correct factual reports remain separate behavioral checks, not proof
that internal representations are preserved during decisions. No honesty claim.

## Later decision, outside this batch

Two competing explanations merit separate designs after this diagnostic: a
controlled payoff manipulation separating represented/recipient objectives from
the total-payoff maximum, and an offset dose comparison addressing unequal strength.
The original payoff cases confound several numerical preferences, so 47/49 alone
does not identify total-payoff maximization. No favorable case selection or new
causal-mechanism claim follows from this exploratory review.
