# Fixed-dose presentation robustness — 2026-10-27

The user requested continued testing after the final dose-grid synthesis. This
separate exploratory scope tests portability of the two already selected offsets.
Dose refinement remains CLOSED. No new strengths, retuning, prompt changes,
training or activation patching are authorized by this protocol.

Question: does the near-match at alpha 0.6875 survive swapping client names and
payoff-column order? Alpha 0.5 is a fixed lower-strength reference, selected for
its previously close payoff performance. Neither is selected using its new
presentation outcomes. The adapter, baseline and full-offset outcomes on these
prompts HAVE already been inspected; this is not a blind holdout or a new task.

Run exactly two GPU jobs, alpha 0.5 and 0.6875, on the unchanged 1,728-case
payoff_name_column1 factorial. Reuse all baseline, adapter and alpha-1 results
from payoff_presentation2. Maximum 3,456 new scored responses plus existing
sentinels and 32 historical checks per job. Three lexical development instances,
two represented clients, two alignments, three row and answer rotations, four
query types and four name/column cells remain. Every cell includes its original
432 prompts; the unchanged cell is an exact same-dose development bridge.

Freeze both runs and the analysis before submitting either. Inherit the complete
inference source from the previous selected-dose snapshot; verify identical source
hashes, model revision, offset artifact, sites, positions and decoding settings.
Gemma-4-31B-it, bf16 SDPA, greedy, thinking disabled, batch 8, max 32 tokens.
Use gpu_pick to inspect availability; retain A100 hardware and prior resource
settings. No live concurrent inference-source edits enter these runs.

Audit all five conditions, reconstruct the factorial and answer keys, verify
native prompts and batch identity across conditions, and check all frozen hashes.
The only old/new source difference allowed is the previously bridged runner
transition recorded in payoff_preferences1 (alpha argument plus the unused
payoff_preferences module; presentation2 already included payoff_diagnostic).
Require every new run's source to equal its same-dose development source and
all 432 unchanged-cell answers AND generated tokens to match that development
run. Require the 32 historical records to match the same dose exactly, including
tokens; do not impose the generic full-strength honesty expectation on alpha 0.5.
Require before/primary/after sentinels. Audit independent runs even if one fails;
any execution discrepancy is reported, never repaired by filtering or retuning.

Report private and factual accuracy, aligned recommendations, opposed recipient
recommendations and valid/dominated counts in every presentation cell. Retain
instance/role/alignment and row/answer strata, all exact adapter disagreements,
paired name/column changes, and full errors with prompts/responses. Compare each
swap's change to the unchanged cell for each condition. Report the matched
context count where recipient recommendation and correct private/factual choices
coexist. Preserve denominators and all cases; do not pool away reversals.

There is no new binary success threshold or significance/equivalence claim.
New private errors would narrow the apparent portability of the offset. Similar
performance would strengthen the simple-offset explanation on these templates.
Neither outcome establishes generalization to independent tasks or a self/other
mechanism. Recommendations are actions, not factual honesty judgments. All
previous failed gates remain failed. Stop after this batch and its analysis.
