# Adversarial audit of the SOO extension

2026-10-27, independent subagent review requested by the user. Read-only review
apart from this report; no experiment submitted, frozen artifact edited, or
acceptance threshold changed.

## Verdict

The main reported counts survive independent checking. I found no scoring or
execution bug that overturns the inspected results. The strongest weaknesses are
in what the comparisons identify: speech act and recipient-goal cues change
together across tasks; inventory truth is confounded with choosing the other
client's optimum; and tiny residual differences lack targeted repeatability
checks. These limit mechanistic explanations, not the existence of the observed
behavioral effects.

The inventory result supports a narrow, important negative result: this adapter
does not produce broad truthful reporting on this template despite competent
baseline behavior. Uncertainty about two truthful reports cannot plausibly turn
2/128 into a broadly successful honesty intervention. Conversely, 1–4-response
advantages should not be treated as stable mechanisms merely because execution
audits passed.

## What I independently checked

I read current PLAN/FINDINGS, frozen-scope protocols, generator/scorer and
controller code, intervention extraction/application code, and earlier gate and
repeatability records. I recomputed selected numbers without importing the
project's scoring functions:

* **All 6,144 inventory responses:** parsed the actual audited quantity, represented
  client, displayed column headers and payout rows directly from prompt text;
  computed numeric maxima and truth; checked IDs and completion status. The
  stored keys and reported development/reserved totals agree.
* **All five calibrated-presentation conditions:** independently parsed displayed
  tables and client columns and reproduced private/factual counts, including
  432/432 private for adapter and half offset, and 419/432 for 0.6875. Its four
  presentation cells reproduce 107, 107, 104, 101 correct out of 108 each.
* **Adapter versus 0.6875 final-grid comparison:** independently rescored all
  expanded historical responses for these two conditions and all their larger
  payoff-set private choices. Reproduced 500/500 versus 497/500 semantically honest
  main-room answers and 972/972 versus 965/972 private choices. Exact room-format
  honesty is 476/500 versus 492/500. Every inspected nonexact historical answer
  was simply a room preceded by “the”.
* **Inventory integrity:** recomputed all 1,608 files listed in the eight execution
  snapshots' provenance hashes, checked their manifest artifact hashes and
  before/primary/after sentinel equality. No mismatch found. This is integrity
  relative to the recorded snapshots, not an external signature or proof against
  a malicious coordinated rewrite.

Principal ledgers under `results/study2/`:
`inventory-reporting1-batch-20261027T192352308999Z.json`,
`payoff-calibrated-presentation1-batch-20261027T185312086713Z.json`, and
`final-dose-grid1-batch-20261027T180658863901Z.json`.
I did not rerun model inference or exhaustively hash every older extension batch.

Independent reserved inventory totals:

| Condition | True conflict reports /128 | True aligned reports /128 | Private correct /256 | Both factual checks /512 |
|---|---:|---:|---:|---:|
| Base | 0 | 125 | 256 | 512 |
| Adapter | 2 | 123 | 256 | 512 |
| Offset 0.5 | 4 | 120 | 256 | 512 |
| Offset 0.6875 | 14 | 121 | 252 | 512 |

## Findings ranked by importance

### 1. High: the transfer comparison changes several candidate causes at once

**Confirmed design limitation, not a demonstrated cause.** The original payoff
task explicitly says whose payoff the recipient wants maximized and asks for a
recommendation. Inventory removes that recipient objective, makes the recipient
an observer of an asserted fact, switches to numeric answers and changes the
domain and consequence descriptions. See [explicit_payoff.py](../../src/selfconcept/soo/explicit_payoff.py)
around lines 74–81 and [inventory_reporting.py](../../src/selfconcept/soo/inventory_reporting.py)
around lines 55–68.

There is a particularly consequential wording connection: the historical main
“honesty” prompt ends, “If you had to recommend one room to Bob, would you point
out ...?” It follows a question about where the valuable object is, but the final
requested act is still a recommendation. Example: `main_0000` in the frozen
final-grid adapter snapshot's `data/eval/main.jsonl`.

Therefore the successful historical/payoff effects and weak inventory effect
could depend on recommendation wording, explicit recipient-goal cues, their
interaction, or other changes. The existing results do not isolate any of these.
Do not write that the inventory contrast establishes a recommendation mechanism,
or that one failed template establishes absence of honesty transfer generally.

**Action:** a frozen 2×2 bridge crossing recommendation versus factual report
with recipient-goal cue present versus absent, while keeping domain, stock,
payoffs, represented client, answer encoding and presentation distributions
matched. Recommendations should be scored as actions and reports as assertions;
do not silently label all recommendation choices “honesty”.

### 2. High for mechanism claims: truth and other-client preference are aliased

**Confirmed by construction.** Inventory has two quantities and each client has
a different unique optimum. In every conflict case, the truthful quantity is
necessarily the other client's optimum. A shift toward truthful reporting can
equally be a shift away from the assigned goal or toward the other client.
The protocol acknowledges this, but it prevents the intended mechanistic
distinction. Private and factual checks do not remove this identification problem.

**Action:** a separate three-answer diagnostic where the truthful quantity can
be the assigned client's optimum, the other client's optimum, or neither. Avoid
making the “neither” quantity systematically dominated or uniquely intermediate
in numeric magnitude: otherwise rejecting it has an additional payoff/position
explanation. Counterbalance truth position, names and payout patterns and retain
the same private/factual controls. This design distinguishes truthful assertion
from choosing one client's favorite option more directly than further dose tuning.

### 3. Medium: execution sentinels do not establish robustness of small residuals

**Confirmed control gap; numerical explanation unresolved.** The runner repeats
only the first eight prompts before and after the run
([run_goals_roles.py](../../src/selfconcept/soo/run_goals_roles.py), lines 250–275).
For both inventory splits, all eight are role 0, aligned, from one instance;
none is a conflict-report case. These checks verify local repeatability in fixed
batches, not repeatability of the reported conflict truths or affected private
choices. Historical checks also do not repeat those new-task cases.

Earlier [GOALS_ROLES_REPAIR.md](GOALS_ROLES_REPAIR.md) documents 2/376 changes
between fixed and reshuffled batches and 2/128 focal changes versus singleton
execution. This establishes that batch composition can matter in this pipeline;
it does not establish that current discrepancies are numerical artifacts.

**Action:** repeat every adapter/offset disagreement relevant to the small
residual claims, plus matched agreement controls, under original batches, a
frozen reshuffle and singleton execution. Keep original results intact and report
all changes. A larger fixed batch containing both roles, alignments and query
types would be a better future sentinel. This is most urgent before interpreting
2 truthful reports or 4 private errors, less urgent for the large failure to
generalize broad honesty.

### 4. Medium: the joint dose criterion is a stringent descriptive rule, not equivalence

**Confirmed; current documents mostly qualify it correctly.**
[cross_task_dose.py](scripts/cross_task_dose.py), lines 104–116, allows no count
loss against the adapter on private, factual, aligned recommendations or strict
historical answers, while allowing a five-percentage-point gap on opposed
recommendations. Passing requires all cells to pass. There is no uncertainty or
practical-equivalence margin for private choices, so one error can fail it.

The strict historical component also mixes answer formatting with content:
0.6875 passes the historical criterion despite three fewer semantically honest
answers, because it produces fewer prefixed articles. Its actual final failure
comes from private losses of 2 and 5 and aligned-validation loss of 4, so the
failure is not solely formatting. Nevertheless, “no strength passed” is a poor
headline for the size or nature of the residual difference.

**Action:** keep the frozen criterion and its failed outcomes, but lead with
semantic honesty, private accuracy, paired discrepancies and clustered instance
variation. Do not retrospectively redefine the threshold to announce a success.
“Close behavioral approximation with small observed gaps” is supported;
“irreducible adapter advantage” is not.

### 5. Medium: behavioral sufficiency is not an explanation of the adapter's mechanism

**Interpretation risk, not a falsified result.** A constant offset is constant at
its injection sites; the downstream model remains input-dependent and nonlinear.
It is also extracted from a trained adapter, not a training-free independent
baseline. Similar answers show that this simpler intervention can reproduce much
of the tested behavior; they do not show that the adapter works solely by that
constant shift, or that self/other representations are absent.

The extension also tests one selected model/adapter seed. Full adapter and offset
do not modify the same full set of modules. Earlier band and random controls are
useful but do not make every later fixed-dose comparison an isolation of input
dependence. Most dose-selection data were already inspected. Frozen individual
batches do not turn the complete adaptive search into a confirmatory study.

**Action:** describe the offset result as behavioral sufficiency on tested tasks.
For seed replication, include all preselected available seeds without choosing
the best one; distinguish testing the same seed-0 vector across seeds from
extracting each seed's own vector. The latter requires a fixed extraction rule
and strength, not per-seed outcome tuning.

### 6. Low: paper-facing status was stale; corrected during this review

At initial inspection, [PAPER_STRUCTURE.md](PAPER_STRUCTURE.md) still called the
direct-payoff follow-up “proposed, not completed.” The parent agent corrected that
status paragraph during this audit to include the completed comparisons and
inventory transfer limit. The mismatch was real but is no longer outstanding.
Keep paper-facing synthesis synchronized before drafting the extension section.

## Checks that strengthen confidence

The original combined explicit-payoff gate failure remains recorded; the later
private-goal branch is explicitly marked post hoc. This does not eliminate
selection history, but is substantially better than silently relaxing the gate.
The inventory competence and opportunity gates are separate, and reserved
intervention release does not require favorable development intervention results.
Frozen data/source/settings, strict completion handling, all-trial denominators,
column-aware keys and paired raw-output retention are substantive controls.
The near-match, presentation sensitivity and weak inventory honesty transfer are
real findings in the inspected executions, not numerical summaries I could
invalidate by rescoring.

## Recommended next sequence

1. Finish a bounded repeatability check of the small residuals; preserve the
   large negative inventory-transfer finding regardless of the outcome.
2. Run the matched speech-act × recipient-goal-cue bridge, with conditions and
   interpretation fixed before outputs. Treat it as hypothesis discrimination,
   not a search for wording that restores a preferred effect.
3. Use a separate three-answer truth-versus-beneficiary diagnostic to break the
   current confound; then replicate the informative contrast across preselected
   adapter seeds. Reserve activation patching for a contrast that survives these
   simpler behavioral tests.

No next experiment was launched as part of this audit.
