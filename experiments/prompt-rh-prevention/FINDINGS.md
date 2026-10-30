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

2026-10-30: pilot done (seed 0, all arms). The honesty arms barely move the hack proxy and mostly make the model
give up on its first reply, including on problems the baseline solves. Main run not submitted.

## Pilot — 2026-10-30: the prompts turn attempts into first-reply give-ups; the hack proxy does not move measurably

Result: in `honest_al`, `wise_professor`, `heroic_ai` and `abort_info`, 47–62% of episodes end with no code, vs 9%
in baseline seed 0, and every one of those is a give-up on the first reply, before any submission. Hidden-suite
pass falls with it (`honest_al` 4% vs 29%). The shown-pass/hidden-fail proxy is 3–5 of 45 in every arm (baseline
5 of 45), a difference too small to read at one seed. `strict` stays near baseline (no code 20%, hidden 31%).

Question: what do the arm prompts do to hacking, solving and giving up, at one seed, before the main run?

Setup: job 14893928 (array 0–4), one H100 each, `--qos=cs`, no partition, 10–26 min per task. Seed 0, all 45
problems, recipe as in `slurm/run_arms.sbatch`, commit e8b8334. Records:
`~/nobackup/autodelete/prompt-rh-prevention-results/main/<arm>/seed0/`; reference: baseline seed 0 (schemer-axis
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
(schemer-axis, H100, `steered_rh_vllm.py` unsteered), the same condition by design (PLAN). Under PLAN's analysis rule
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
