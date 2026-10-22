# Selection and confirmation protocol (frozen 2026-09-18)

Written in response to audit items 2 and 9 (`AUDIT.md`), which showed that the study's
transfer claims mix discovery, prospective prediction and post-hoc revision, and that the
n=50 pilots are prefixes of the n=250 confirmation files. Everything before this date stays
exploratory by definition; this protocol governs claims made from here on.

Nothing here restricts exploration. Search as widely as the compute allows. The rule is only
that a result reported as *confirmed* must have been produced by the path below, and a result
produced any other way is reported as *exploratory*.

## 1. Development and confirmation data are disjoint

**In-distribution.** Each scenario file holds 250 examples, indices 0–249, and the n=50 pilots
are the first 50. Therefore:

- Development (layer, α, token mode, position mode, orientation, prompt format, output channel):
  indices **0–49** only, any number of looks.
- Confirmation: indices **50–249** (n=200), reported separately from the pilot. Do not quote a
  single n=250 number that pools them; it contains the selection set.
- Both orientations (original + mirrored) always, for both splits.

**Apollo roleplaying.** 371 scenarios, 89 distinct speaker roles.
`data/apollo_rp_role_overlap.json` splits them by whether the role appears in the training
pairs: 125 overlap / 246 disjoint. A transfer claim reports the **disjoint** 246 as primary and
the overlap 125 as a secondary observation. Role-disjoint is not situation- or
template-disjoint; say so, and do not call it out-of-distribution without a family-level holdout.

**Family-level holdout.** A claim that SOO generalizes *beyond the trained situations* requires
a scenario family never inspected during development of the pair set. Neither the Apollo
roleplaying nor the insider scaffold qualifies once pairs have been written against them.
Build or reserve one before making that claim.

## 2. What must be fixed before the confirmation run

Write these into the round's slurm script or PLAN entry **before submitting the generation
job**, in a commit that precedes the results:

1. The predictor and its threshold, stated as an arithmetic rule over quantities already
   measured (e.g. "direction-specific = −v within X points of baseline **and** the three random
   seeds' mean within Y points of baseline").
2. The exact cells: model, layer, α, orientation, seeds, n.
3. The outcome metric and its denominator, including how missing/unparsed records are handled.
4. The success criterion, as a signed, quantified prediction per cell.
5. Which comparisons are confirmatory (a short list, multiplicity-adjusted) and which are
   exploratory (everything else, reported without significance claims).

"Partial transfer" is not admissible unless step 1 defines it numerically in advance. Round 3's
tally moved from 5/7 to "8/9" precisely by introducing a graded category after the misses.

## 3. Analysis rules

- **Paired by default.** Conditions share example IDs, so use exact McNemar plus a paired item
  bootstrap (`scripts/paired_tests.py`). Independent-sample Fisher/z tests are not reported.
- **Direction specificity is a separate test.** Significance against baseline does not establish
  that a vector's effect is about *its direction*. Report the contrast against the random-vector
  controls as its own line; if that contrast is not significant, the effect is not
  direction-specific, whatever the baseline comparison says. (Qwen3.8 L31 roleplaying:
  p=0.0073 vs base, p=0.18 vs random.)
- **Three random seeds minimum**, reported as a distribution, not a single control.
- **No equivalence claims from a null.** "No significant cost" on n=40 is an underpowered test.
  To claim an intervention is free, prespecify a noninferiority margin and show the paired CI
  excludes it. Otherwise say "we did not detect a cost at n=40, CI [a, b]".
- **Repeated seeds, doses and orientations are not independent units.** Do not pool them into one
  contingency table; cluster or report per-cell.
- **Missing evidence is not an outcome.** Any label that can be produced by truncation,
  refusal or a parse failure must separate completion from behaviour, with the denominator
  stated (the sandbagging repair is the worked example: see `scripts/reparse_sandbagging.py`).

## 4. Judges

No judged metric is confirmatory until that judge has (a) an independent-family cross-grade and
(b) a blinded human check on a stratified sample. `scripts/make_apollo_validation_sample.py`
builds both for the roleplaying judge. Report agreement and Cohen's kappa on the binary call,
stratified by condition and response length. The insider and sandbagging paths need the same
treatment before their numbers carry a claim.

## 5. Provenance

Every summary records: model revision, code revision, adapter/vector path and hash, dataset
hash, `SOO_CHAT_KWARGS`, generation budget, decoding settings, and per-record completion status.
Output tags are unique per condition; never reuse a tag across settings. (Not yet implemented in
`evaluate.py` — scheduled for the next window when the job queue is empty, so one round's
summaries do not straddle two schemas.)

## 6. Reporting vocabulary

| word | licensed by |
|---|---|
| observed / exploratory | any search, any number of looks |
| replicated | same cell, ≥3 seeds, paired, on confirmation data |
| predicted | §2 rule committed before the run, evaluated on cells never used to build it |
| direction-specific | significant against the random-control distribution, not just baseline |
| free / capability-preserving | prespecified noninferiority margin excluded by the paired CI |
| transfers | confirmation data + validated judge + family-level holdout named |

Anything else is described, not claimed.
