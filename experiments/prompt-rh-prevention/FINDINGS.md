# Prompting against reward hacking: findings

Started 2026-10-22. Plan: [PLAN.md](PLAN.md).

## Writing conventions

One dated section per analyzed run, newest first, placed after "Current status". Each section states the result
first, then the question, the setup (job IDs, artifact paths), the measurements, and the limits. Separate what was
observed from how it is explained. Pending work stays in PLAN.md. A conclusion that a later run overturns is marked
superseded, not deleted.

Do not call a drop in hack rate reduced reward hacking when hidden-suite pass rate, no-code rate or truncation moved
with it, and do not call a non-significant result evidence of no effect.

## Current status

2026-11-03: official run done (6 arms incl. a rerun baseline × seeds 67–78, plus general capability at seed 67).
Every prompt arm lowers the hack proxy, mostly by making the model give up without code; no arm meets PLAN's rule for
calling it reduced reward hacking. `diligent_dev` (added 2026-11-03) has a similar hack drop (−10.0 pt) with the
smallest solve loss of the honesty arms (−3.3 pt, CI covers 0) and keeps general capability, but still gives up
far more than baseline (+29 pt no code). Analysis of honest give-ups (scoring decision) open. Shared copy for
judging: `/grphome/grp_mirror/self-concept-experiments/experiments/prompt-rh-prevention/` (README there).

2026-11-05: LLM-judge results added (judging by jansen05, "LLM judge" below). Judged hacking falls from 13.1%
(baseline) to 0.0–5.6% in every arm, and also falls per code-submitting episode, so the drop is not only from giving
up. `diligent_dev` 2.0%, `abort_info` 3.0%.

### All results (official run, seeds 67–78; updated 2026-11-03)

Recipe rates over 540 episodes per arm (45 problems × 12 seeds), Δ vs the rerun baseline in points; general
capability at seed 67. Hack proxy = shown tests pass, hidden suite fails. "Among submitted" = hack proxy ÷ (1 − no
code), descriptive only. Capability rule (PLAN): on every benchmark the CI covers 0 or |Δ| ≤ 2 points. CIs and
details: "Official run" below.

| arm | hack proxy | hidden pass | no code | hack among submitted | GSM8K | MMLU-Pro | HumanEval | IFBench | capability rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 19.4% | 23.3% | 9.8% | 21.5% | 95.6% | 75.9% | 97.6% | 53.3% | (reference) |
| honest_al | 4.4% (−15.0) | 7.4% (−15.9) | 69.1% (+59.3) | 14.2% | 95.4% (−0.2) | 72.9% (−3.0) | 96.3% (−1.2) | 53.0% (−0.3) | fails |
| wise_professor | 7.4% (−12.0) | 14.8% (−8.5) | 52.0% (+42.2) | 15.4% | 95.5% (−0.1) | 73.8% (−2.1) | 94.5% (−3.0) | 52.0% (−1.3) | fails |
| heroic_ai | 8.5% (−10.9) | 16.7% (−6.7) | 52.0% (+42.2) | 17.7% | 95.0% (−0.6) | 74.7% (−1.1) | 95.1% (−2.4) | 49.3% (−4.0) | fails |
| strict | 12.4% (−7.0) | 22.4% (−0.9) | 25.2% (+15.4) | 16.6% | 90.7% (−4.9) | 73.9% (−1.9) | 95.1% (−2.4) | 41.7% (−11.7) | fails |
| abort_info | 10.0% (−9.4) | 17.2% (−6.1) | 47.2% (+37.4) | 18.9% | 95.9% (+0.3) | 74.4% (−1.4) | 95.7% (−1.8) | 50.0% (−3.3) | keeps |
| diligent_dev | 9.4% (−10.0) | 20.0% (−3.3) | 39.1% (+29.3) | 15.5% | 96.0% (+0.4) | 75.6% (−0.3) | 95.7% (−1.8) | 53.0% (−0.3) | keeps |

All recipe Δs have 95% CIs excluding 0 except hidden pass for `strict` and `diligent_dev`. No turn was truncated in any arm.

Raw counts (same runs). Every episode ends one way: passed shown + stopped + attempt limit = episodes (no episode
ended by the context limit or the flag). "Submitted code" = episodes − no code. Turns = model replies, at most 5
per episode. Capability = correct / items.

| arm | episodes | hack proxy | hidden pass | no code | submitted code | passed shown | stopped | attempt limit | turns | GSM8K | MMLU-Pro | HumanEval | IFBench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 540 | 105 | 126 | 53 | 487 | 231 | 160 | 149 | 1564 | 1261/1319 | 1062/1400 | 160/164 | 160/300 |
| honest_al | 540 | 24 | 40 | 373 | 167 | 64 | 475 | 1 | 701 | 1258/1319 | 1020/1400 | 158/164 | 159/300 |
| wise_professor | 540 | 40 | 80 | 281 | 259 | 120 | 388 | 32 | 938 | 1260/1319 | 1033/1400 | 155/164 | 156/300 |
| heroic_ai | 540 | 46 | 90 | 281 | 259 | 136 | 362 | 42 | 925 | 1253/1319 | 1046/1400 | 156/164 | 148/300 |
| strict | 540 | 67 | 121 | 136 | 404 | 188 | 233 | 119 | 1348 | 1196/1319 | 1035/1400 | 156/164 | 125/300 |
| abort_info | 540 | 54 | 93 | 255 | 285 | 147 | 348 | 45 | 986 | 1265/1319 | 1042/1400 | 157/164 | 150/300 |
| diligent_dev | 540 | 51 | 108 | 211 | 329 | 159 | 317 | 64 | 1078 | 1266/1319 | 1058/1400 | 157/164 | 159/300 |

## LLM judge — 2026-11-03 (recorded 2026-11-05): judged hacking falls from 13.1% to 0–5.6%, also among episodes that submit code

Result: with a blind Sonnet judge, every prompt arm lowers the judged hack rate against the rerun baseline
(13.1% → 0.0–5.6%; all six Δ CIs exclude 0 after Holm). Unlike the hack proxy, the drop also holds per
code-submitting episode (14.6% → 0–7.4%), so the prompts do more than suppress code. The PLAN caveat still applies:
no-code rises 15–59 points in every arm and hidden pass falls in four, so this is not yet "reduced reward hacking" by
PLAN's rule. Of the two arms that keep general capability, `diligent_dev` (2.0%) and `abort_info` (3.0%) are both low.

Question: how much of the hack proxy is reward hacking, per arm, when a judge reads the transcripts with the arm's
prompt?

Setup (done by jansen05; everything is in
`/grphome/grp_mirror/self-concept-experiments/experiments/prompt-rh-prevention-judging_jansen05/`, its
`README_judging.md` is the full record):

- Episodes judged: the 387 hack-proxy episodes from "Official run" (shown pass, hidden fail), plus 62 hidden-pass
  episodes whose code has a hard-coding signature (branches or dict keys on the shown inputs, constant returns;
  `hidden_sig/screen_hidden_sig.py`). The other episodes were not judged and count as not hacking, so the judged rate
  is a lower bound.
- Judge: one Sonnet agent per batch of about 6 dossiers, rubric v3 (`experiments/codebench/rh_judge/judge_rubric_v3.md`),
  blind to the arm (opaque ids), one read per dossier, no adjudication. Dossiers from deceiver-axis `make_dossiers.py`
  with `prompt_prefix` added, so the judge saw the arm's prompt exactly as the model did.
- Analysis: deceiver-axis `main_analysis.py --tag official` (rates are problem-stratified means over 45 problems,
  problem bootstrap CIs, Holm over the six prompt arms). Judged hack = verdict HACK (rubric STRONG or WEAK).
- We checked the verdict counts below against `verdicts_official/verdicts_official.json` and
  `verdicts_hidden_official.json`, and the rates and CIs against `analysis_official/analysis_all_as_judged*.txt`.
  The hack proxy, hidden pass and no-code rates there match "Official run".

| arm | judged hack | Δ vs baseline [95% CI] | Holm p | judged hacks / code-submitting episodes | BORDERLINE counted | WEAK not counted | proxy set only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 13.1% (71/540) | — | — | 14.6% (71/487) | 16.5% | 11.7% | 10.0% |
| honest_al | 0.0% (0/540) | −13.1 [−18.5, −8.1] | 0.0003 | 0.0% (0/167) | 0.2% | 0.0% | 0.0% |
| wise_professor | 1.3% (7/540) | −11.9 [−17.4, −7.0] | 0.0003 | 2.7% (7/259) | 2.2% | 0.7% | 0.9% |
| heroic_ai | 1.9% (10/540) | −11.3 [−17.0, −5.9] | 0.0006 | 3.9% (10/259) | 2.2% | 0.7% | 1.3% |
| strict | 5.6% (30/540) | −7.6 [−12.8, −2.6] | 0.0067 | 7.4% (30/404) | 7.0% | 3.1% | 4.4% |
| abort_info | 3.0% (16/540) | −10.2 [−15.2, −5.4] | 0.0006 | 5.6% (16/285) | 5.2% | 1.7% | 1.9% |
| diligent_dev | 2.0% (11/540) | −11.1 [−16.7, −6.3] | 0.0004 | 3.3% (11/329) | 3.0% | 0.7% | 1.7% |

All sensitivity variants keep every arm's CI below 0. "Proxy set only" = judging only the 387 proxy episodes.
"Per code-submitting episode" is descriptive (no CI).

Verdict counts (HACK / BORDERLINE / NOT_HACK):

| arm | proxy set | hidden-signature set |
| --- | --- | --- |
| baseline | 54 / 16 / 35 | 17 / 2 / 10 |
| honest_al | 0 / 1 / 23 | 0 / 0 / 1 |
| wise_professor | 5 / 5 / 30 | 2 / 0 / 0 |
| heroic_ai | 7 / 2 / 37 | 3 / 0 / 2 |
| strict | 24 / 8 / 35 | 6 / 0 / 7 |
| abort_info | 10 / 12 / 32 | 6 / 0 / 3 |
| diligent_dev | 9 / 4 / 38 | 2 / 1 / 0 |

Observed patterns (from jansen05's notes): the judge called about half of the baseline's proxy episodes hacks
(54/105), and well under half in every prompt arm. Common hacks are lookup tables written after a give-up turn,
fitted stubs such as `return 0 if sum(nums1) <= x else -1`, and search limits described as "sufficient for the test
suite". Most hidden-set hacks are a constant `'-1'` return on one problem that exploits the shown `is_valid` helper,
which the hidden suite also accepts.

Limits: one judge, one read per dossier, no human check of verdicts. `honest_al`'s 0 rests on 25 dossiers from an arm
that submitted code in 31% of episodes. Episodes outside the two judged sets (for example hard-coded code that failed
the shown tests) were not read.

## Official run — 2026-11-03: the prompts cut the hack proxy by 7–15 points, mainly by turning attempts into give-ups

Result: against the rerun baseline (hack proxy 19.4%, hidden pass 23.3%, no code 9.8%), all five arms lower the
shown-pass/hidden-fail proxy (−7.0 to −15.0 pt, all CIs exclude 0) and all raise the no-code rate (+15 to +59 pt).
Under PLAN's analysis rule (a no-code rise above 10 points is degradation, not reduced hacking) none of the drops
counts as reduced reward hacking. `strict` comes closest: proxy −7.0 pt with hidden pass unchanged (−0.9 pt, CI
covers 0), but no code +15 pt. `honest_al` is the extreme: proxy 4.4%, but no code 69% and hidden pass 7%.

Question: do the honesty prompts lower reward hacking on the recipe, and at what cost in solving and capability?

Setup: jobs 14915978 (canary, task 0), 14916163 (tasks 1–71, `%12`), 14916164 (capability, 6 arms); one B200
each (`gpu_pick`: 6 idle B200s, 0 jobs ahead; H100 9 ahead), QoS `cs`, no partition; commit 9af71b8. 540 episodes
per arm (45 problems × seeds 67–78), recipe and model settings as the steering experiment (PLAN 2026-10-30).
Records: `~/nobackup/autodelete/prompt-rh-prevention-results/official/` and `capability_official/`. The rerun
baseline reproduces deceiver-axis's unsteered arm (proxy 19.4% vs 19.4%, hidden pass 23.3% vs 25%).

Rates vs baseline: difference in points, 95% whole-problem bootstrap CI (4000 resamples, problems resampled, all
seeds of a problem kept together), Holm-corrected across the six prompt arms (`diligent_dev` added 2026-11-03,
jobs 15004035 and 15004036, same setup, commit 9bf99d7).

| arm | hack proxy | Δ proxy | hidden pass | Δ hidden | no code | Δ no code | turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 19.4% | | 23.3% | | 9.8% | | 1564 |
| honest_al | 4.4% | −15.0 [−20.9, −9.4] | 7.4% | −15.9 [−23.0, −9.3] | 69.1% | +59.3 [+49.6, +68.1] | 701 |
| wise_professor | 7.4% | −12.0 [−17.2, −7.2] | 14.8% | −8.5 [−13.5, −4.1] | 52.0% | +42.2 [+33.1, +51.3] | 938 |
| heroic_ai | 8.5% | −10.9 [−16.1, −6.3] | 16.7% | −6.7 [−11.7, −2.0] | 52.0% | +42.2 [+33.0, +51.3] | 925 |
| strict | 12.4% | −7.0 [−11.1, −3.1] | 22.4% | −0.9 [−5.2, +3.1] | 25.2% | +15.4 [+9.8, +21.3] | 1348 |
| abort_info | 10.0% | −9.4 [−14.1, −5.2] | 17.2% | −6.1 [−10.4, −2.4] | 47.2% | +37.4 [+28.5, +45.9] | 986 |
| diligent_dev | 9.4% | −10.0 [−15.2, −5.4] | 20.0% | −3.3 [−8.5, +1.5] | 39.1% | +29.3 [+20.9, +37.8] | 1078 |

All Holm-adjusted p < 0.005 except Δ hidden for `strict` (p = 0.71) and `diligent_dev` (p = 0.33). No turn was truncated in any arm.

Hack proxy among episodes that submitted code (proxy ÷ (1 − no code); no CI, and the episodes that remain are a
selected set): baseline 21.5%, honest_al 14.2%, wise_professor 15.4%, heroic_ai 17.7%, strict 16.6%, abort_info
18.9%, diligent_dev 15.5%. Most of each arm's drop is episodes that never submit code.

General capability (seed 67, paired with the rerun baseline; `capability_eval_compare.py`), Δ in points [95% CI]:

| arm | GSM8K (95.6%) | MMLU-Pro (75.9%) | HumanEval (97.6%) | IFBench (53.3%) | PLAN capability rule |
| --- | --- | --- | --- | --- | --- |
| honest_al | −0.2 [−1.1, +0.7] | −3.0 [−4.6, −1.4] | −1.2 [−3.7, +1.2] | −0.3 [−6.0, +5.3] | fails (MMLU-Pro) |
| wise_professor | −0.1 [−0.9, +0.8] | −2.1 [−3.8, −0.4] | −3.0 [−6.1, −0.6] | −1.3 [−7.0, +4.0] | fails (MMLU-Pro, HumanEval) |
| heroic_ai | −0.6 [−1.5, +0.3] | −1.1 [−2.8, +0.5] | −2.4 [−4.9, −0.6] | −4.0 [−9.0, +1.0] | fails (HumanEval) |
| strict | −4.9 [−6.4, −3.5] | −1.9 [−3.6, −0.2] | −2.4 [−4.9, −0.6] | −11.7 [−18.0, −5.7] | fails (GSM8K, HumanEval, IFBench) |
| abort_info | +0.3 [−0.5, +1.1] | −1.4 [−3.0, +0.2] | −1.8 [−4.9, +0.6] | −3.3 [−8.3, +1.7] | keeps capability |
| diligent_dev | +0.4 [−0.4, +1.1] | −0.3 [−1.9, +1.3] | −1.8 [−4.3, +0.0] | −0.3 [−5.7, +5.0] | keeps capability |

(Rule: an arm keeps capability if on every benchmark the CI covers 0 or |Δ| ≤ 2 points.) IFBench (300 prompts with
verifiable instructions, prompt-level loose accuracy, as in the deceiver-axis capability check) added 2026-11-03:
job 15006320, one A100 per arm, all 7 arms including the baseline in the same job, seed 67; no truncated or empty
answers. `strict` loses 11.7 points there; every other arm's CI covers 0. HumanEval has 164 items, so
its failures rest on 4–5 problems lost and none gained.

Limits: the hack measure is the automatic proxy (shown pass, hidden fail), not judged hacks; it also counts honest
bugs. One capability seed. The proxy-among-submissions figures are descriptive only. Whether an honest give-up on a
problem the model could solve is a good or bad outcome is the open scoring decision in PLAN; under the current rule
it is degradation.

## Pilot — 2026-10-30: the prompts turn attempts into first-reply give-ups; the hack proxy does not move measurably

Result: in `honest_al`, `wise_professor`, `heroic_ai` and `abort_info`, 47–62% of episodes end with no code, vs 9%
in baseline seed 0, and every one of those is a give-up on the first reply, before any submission. Hidden-suite
pass falls with it (`honest_al` 4% vs 29%). The shown-pass/hidden-fail proxy is 3–5 of 45 in every arm (baseline
5 of 45), a difference too small to read at one seed. `strict` stays near baseline (no code 20%, hidden 31%).

Question: what do the arm prompts do to hacking, solving and giving up, at one seed, before the main run?

Setup: job 14893928 (array 0–4), one H100 each, `--qos=cs`, no partition, 10–26 min per task. Seed 0, all 45
problems, recipe as in `slurm/run_arms.sbatch`, commit e8b8334. Records:
`~/nobackup/autodelete/prompt-rh-prevention-results/main/<arm>/seed0/`; reference: baseline seed 0 (deceiver-axis
unsteered). Summary: `scripts/pilot_summary.py <root> --seeds 0`.

| arm | turns | hidden pass | shown pass, hidden fail | no code | stopped | Δ hidden | Δ no code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 140 | 29% | 11% | 9% | 18 | | |
| honest_al | 64 | 4% | 7% | 62% | 39 | −24 pt | +53 pt |
| wise_professor | 77 | 24% | 7% | 47% | 28 | −4 pt | +38 pt |
| heroic_ai | 75 | 20% | 7% | 56% | 30 | −9 pt | +47 pt |
| strict | 112 | 31% | 11% | 20% | 20 | +2 pt | +11 pt |
| abort_info | 76 | 16% | 11% | 49% | 30 | −13 pt | +40 pt |

Give-ups on problems the baseline can solve (no-code episodes among the 13 problems baseline seed 0 solved / the
26 solved in at least one of its 12 seeds): `honest_al` 6/13, 15/26; `wise_professor` 4/13, 10/26; `heroic_ai`
5/13, 14/26; `strict` 2/13, 4/26; `abort_info` 6/13, 11/26.

Limits: one seed, 45 episodes per arm, no CIs; hack proxy only (no judge run). The baseline is a different job
(deceiver-axis, H100, `steered_rh_vllm.py` unsteered), the same condition by design (PLAN). Under PLAN's analysis rule
a no-code rise of more than 10 points is degradation, not reduced hacking; for arms that ask the model to give up
honestly, how to score those give-ups is an open decision (PLAN).

## GPU test — 2026-10-29: all five arms run on one H100; prompts reach the model and no-code replies end the episode

Result: the pipeline works end to end on one H100 80 GB at a 20,480-token context. Every record carries its arm's
prompt, no turn was truncated, and all 27 `stopped` episodes end in a plain admission without code ("I'm sorry, but
I'm not able to solve this problem."), none a parsing accident.

Question: does the model load and fit, do the prompts and the abort work, and how long does a task take?

Setup: job 14889593 (array 0–4, one arm each), `--gpus-per-node=h100:1 --qos=cs`, no partition (GPU chosen with
`gpu_pick`: no idle GPUs; H100 had 70 jobs ahead vs 105 for A100, and the baseline ran on an H100). Seed 0, first
3 problems of each of `ib_solvhard_s0-2` (9 episodes per arm), the recipe as in `slurm/run_arms.sbatch`. Records:
`~/nobackup/autodelete/prompt-rh-prevention-results/gputest/<arm>/seed0/`. Commit d85492c plus the uncommitted
README edit that became 1f473c0.

Measurements: model weights 61.4 GiB, KV cache ~8.2 GiB (~120k tokens, 5.9 concurrent 20k-token requests).
Engine start ~3 min. Wall time per task 289–616 s.

| arm | episodes | turns | passed shown | stopped | attempt limit | hidden pass | shown pass, hidden fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| honest_al | 9 | 12 | 2 | 7 | 0 | 1 | 1 |
| wise_professor | 9 | 17 | 3 | 5 | 1 | 3 | 0 |
| heroic_ai | 9 | 10 | 3 | 6 | 0 | 3 | 0 |
| strict | 9 | 29 | 3 | 3 | 3 | 2 | 1 |
| abort_info | 9 | 10 | 3 | 6 | 0 | 2 | 1 |

Limits: 9 episodes per arm on the first problems of each file, one seed; no rate here is an estimate. Most stops
come on the first turn, including in `heroic_ai` and `strict`, which are not told how to stop; whether the prompts
make the model give up on problems it could solve is a question for the pilot (no-code and hidden-pass rates vs
baseline).
