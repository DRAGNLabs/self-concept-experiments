# Schemer axis and reward hacking: findings

Started 2026-10-02. Plan: [PLAN.md](PLAN.md).

## Writing conventions

One dated section per analyzed run, newest first, placed after "Current status". Each section states the result
first, then the question, the setup (job IDs, artifact paths, vector and cohort hashes), the measurements, and the
limits. Separate what was observed from how it is explained. Pending work stays in PLAN.md. A conclusion that a later
run overturns is marked superseded, not deleted.

Do not call a direction "the schemer direction" until it has passed the Phase 1 validation checks. Do not call a
drop in hack rate reduced reward hacking when hidden-suite pass rate, no-code rate or truncation moved with it. Do
not call a result that is not distinguishable from random directions an association with this direction, and do not
call a non-significant result evidence of no effect.

## Current status

Nothing has been run. The plan was written on 2026-10-02; the next step is the role files and the Phase 1
extraction job.

| Question | Status | Answer so far |
|---|---|---|
| 1. Is there a usable schemer direction in gpt-oss-120b? | not started | — |
| 2. Does it separate hack from non-hack episodes? | not started | — |
| 3. Does shifting or capping along it change the hack rate? | not started | — |

## Runs

None yet.

## Background: the baseline this study compares against

Not a result of this study. From the random-vector control of 2026-10-01 (branch `random-vectors-rh`, output in
`~/nobackup/autodelete/rh-random-control-v3/full/analysis/`): Assistant-Axis projection vs reward hacking on
gpt-oss-120b at layer 17, 824 transcripts (189 HACK, 635 unflagged), 41 problems, hacks compared with non-hacks of
the same problem, run family and turn, against 256 isotropic random directions.

| Region | Pairs | AA AUROC | Random 2.5 / 50 / 97.5% | Tail fraction | Token-count AUROC | Residual-norm AUROC |
|---|---:|---:|---|---:|---:|---:|
| prompt | 868 | 0.525 | 0.430 / 0.499 / 0.571 | 0.54 | 0.487 | 0.497 |
| cot | 867 | 0.696 | 0.311 / 0.504 / 0.691 | 0.044 | 0.442 | 0.565 |
| final | 868 | 0.714 | 0.145 / 0.504 / 0.849 | 0.29 | 0.544 | 0.116 |

AUROC above 0.5 means hack turns project higher on the Assistant Axis. In the final region the association is not
distinguishable from random directions, and the norm of the mean residual alone separates the labels. In the CoT
region it is at the edge of the random-direction range.
