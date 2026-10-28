# Payoff preferences and offset strength — follow-up 1

## Question and scope

User authorization, 2026-10-27 Mountain: continue by separating competing payoff
preferences and checking whether offset strength explains its additional errors.
This is exploratory work after all earlier results, not untouched validation.
The failed original combined gate stays failed. No honesty or internal self/other
mechanism claim, prompt search, training, or downstream automatic experiments.

Two bounded branches, frozen before outputs, maximum **9,936 scored generations**:
five original-development offset runs (5×432 = 2,160), and three numerical diagnostic
conditions (3×2,592 = 7,776). Freeze all eight execution snapshots first. Submit the
five dose runs and numerical baseline. Analyze endpoints and baseline before
releasing the two numerical interventions. Stop release on execution, repeat,
endpoint-bridge, or baseline feasibility failures; record all outcomes.

## A. Strength tradeoff on unchanged development prompts

Reuse all original 432 development prompts, keys, batches and order. Run offset
alpha **0, 0.25, 0.5, 0.75, 1.0**, using the same extracted direction at the same
sites/positions. Reuse original baseline and full-adapter outputs for comparisons.
The only inference change adds an explicit alpha argument, default 1.0; record it
in manifests and hook metadata. Preserve bf16 SDPA, greedy decoding, batch 8,
thinking disabled, 32-token cap, pinned Gemma revision and all intervention artifacts.

Require alpha 0 to reproduce every saved baseline answer and generated token,
and alpha 1 every saved offset answer/token, on all 432 prompts and 32 historical
regression prompts. Exact prompt tokens, batches, settings, package versions and
unchanged source modules must match. Permitted source differences are only the
runner's alpha plumbing and the inference-unused presentation/numerical generator
modules added since the original execution snapshots.
Sentinel repeats must pass for every dose. Historical-label changes at intermediate
doses are outcomes: do not require a weak/zero offset to retain full-dose honesty
on the old task. Require complete historical generations and report labels at all
doses; endpoint identity supplies the full-dose/zero checks.

Report each dose's private/factual accuracy, opposed recipient-optimal recommendations,
aligned recommendation errors, historical labels, and instance/role/order cells.
Plot private accuracy against opposed recipient-optimal recommendation rate, with
baseline and full adapter. Show effect retention as
(offset recipient count − baseline recipient count)/(adapter count − baseline count),
without clipping. No cherry-picked dose or fitted optimum. Descriptively flag
whether any tested nonzero dose has all 108 private and 216 facts correct while
exceeding baseline opposed recipient choices. That is weaker than matching the
adapter; report the actual recommendation rates. If doses do not match adapter
communication effectiveness, do not claim a matched-strength comparison or rule
out a strength/tradeoff explanation. No interpolation or new dose selection.

## B. Numerical preferences with distinct predictions

Derive prompts from the same three frozen lexical development instances. Change
only payoff numbers; client goals, recommendation consequences, clauses, expected
client-optimal answers, row/answer orders and all query types remain unchanged.
Each instance crosses six feasible patterns of which client's optimum maximizes
the total, the minimum individual payoff, and the largest single payoff. For
client-0-best option A and client-1-best option B, base payoff matrices are:

| Profile | A (client 0, client 1) | B (client 0, client 1) | Total winner | Minimum winner | Largest-single winner |
|---|---|---|---|---|---|
| 0 | (12,5) | (2,10) | A | A | A |
| 1 | (10,6) | (1,12) | A | A | B |
| 2 | (14,1) | (4,10) | A | B | A |
| 3 | (10,4) | (1,14) | B | A | B |
| 4 | (12,1) | (6,10) | B | B | A |
| 5 | (10,2) | (5,12) | B | B | B |

The third option gives (0,0) and is strictly dominated. Multiply every payoff by
1, 2 or 3 in the corresponding lexical instance; this preserves all rankings.
The sum of an option's minimum and maximum equals its total, so the total winner
cannot disagree with both other winners. These are all six feasible combinations,
not an independent 2×2×2 design. Magnitudes/scaling and names are not separately
identified by these three reused templates.

Each profile includes all original 432 conditions (both roles, both alignments,
three row rotations, three answer rotations, four queries): **2,592 per condition**.
Run baseline, full adapter and full-strength offset only. This design compares
candidate behavioral rules; it does not establish an internal optimizer. Original
random-control and band results remain documented but are not newly run here.

Baseline feasibility requires >=95% valid and >=90% correct for private/fact0/fact1,
globally, role×alignment, row×answer order, and each numerical profile (120 checks).
Recommendations are descriptive and cannot release or stop this branch. All fixed-
batch sentinels, inactive-hook and historical baseline checks remain required.
Failure stops without changing wording or thresholds. Alpha endpoint bridges must
also pass before the adapter/offset numerical runs are released.

Analyze all trials, including factual failures. Report rule matches by condition,
query, profile, lexical instance, role/alignment, and row/answer order. Compare
total versus minimum, total versus largest-single, and minimum versus largest-single
specifically where their predictions disagree, retaining denominators and all errors.
Also compare assigned-client and recipient optima. Keep private/factual checks with
matched recommendations. Paired profiles are controlled numerical changes but their
values co-vary; do not claim a uniquely manipulated utility coefficient. No prompt-
level significance tests: rotations/reused templates are not independent tasks.

## Verification, execution and stopping

Check unique IDs, complete factorial counts, strict dominance, unique goal and rule
keys, exact unchanged non-payoff clauses, and consistent displayed values independently.
Native-tokenizer span validation precedes GPU submission. Freeze code, data, tests,
protocol, preflight, intervention assets and resource-picker output. Retain original
closed ledgers/snapshots unchanged. Audit the source-module differences explicitly.
Persist all receipts and status transitions. Controller exceptions record inspection-
needed status and never silently waive checks or resubmit. Analyze numerical
interventions only after both complete; no additional jobs follow that analysis.
