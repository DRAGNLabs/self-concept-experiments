# CoT-ness/deception experiment status

## Current follow-up — submitted 2026-09-23

**EvilGenie caveat (2026-09-24).** The shared code harness showed the workspace files through a 3000-char middle truncation, so the model never saw the middle of test.py (all 138 problems) and saw about a third of the visible tests on the median problem. The pilot, repair and the running expansion (frozen snapshots) all inherit this; their EvilGenie labels measure behaviour on an under-specified task and should not be read as reward-hacking rates. Fixed in `src/selfconcept/codebench/harness.py` on 2026-09-24 (details in experiments/soo/FINDINGS.md, last section); future EvilGenie runs need a new snapshot.

All four repair runs finished and passed their completion/alignment gates: 94/96 task outputs complete, two truncated. Coding completion was 4/4 for both Gemma models and Qwen, and 3/4 for Muse. All four expanded runs are active, about 20–23 hours into execution (checked 2026-09-24 21:19 UTC); judges remain queued behind generation.

Expanded tasks recorded: **1364/1,472 (92.7%)**, comprising 1356 complete, 7 truncated, and 1 with no final answer. All 1,280 ordinary tasks have recorded outcomes; the remaining work is coding tasks, which can take substantially longer per task.

| Model | Recorded tasks / planned | Complete | Truncated | No final |
|---|---:|---:|---:|---:|
| gemma4-12b | 340 / 368 | 336 | 4 | 0 |
| gemma4-31b | 348 / 368 | 348 | 0 | 0 |
| qwen38-27b | 342 / 368 | 339 | 3 | 0 |
| muse-30b | 334 / 368 | 333 | 0 | 1 |

Generated-text token alignment is unavailable for 8 recorded expanded generations (seven Qwen, one Muse); their generated-CoT/final region scores are omitted. Their prompt scores remain available. Repair alignment was exact throughout. No new judged correlation conclusion is available yet.

- Repair: 24 tasks/model, 96 total, on previously inspected examples.
- Conditional expansion: 368 additional tasks/model, at most 1,472 total; disjoint from the pilot and repair.
- Seeded sampling at temperature 1.0, top_p 0.95, top_k 64; 8,192 ordinary / 32,768 coding tokens.
- Frozen validated probes reused unchanged. Kimi excluded after finding unsupported reasoning markers in the original setup.
- Expansion requires a complete repair manifest, valid probe, all expected tasks, at least 16/20 complete ordinary and 3/4 complete coding outputs, first-turn measurements for every task and exact token alignment. No deception outcome or correlation enters this gate.
- Each judge handles repair and expanded outputs separately. New judged correlation results are not available yet.

| Model | Repair job | Gated expansion job | Judge job |
|---|---:|---:|---:|
| gemma4-12b | 13873906 | 13873910 | 13873911 |
| qwen38-27b | 13873907 | 13873912 | 13873913 |
| gemma4-31b | 13873908 | 13873915 | 13873916 |
| muse-30b | 13873909 | 13873917 | 13873918 |

[Frozen launch and selected IDs](results/followup-20260923T203229Z/launch.json) · [Protocol and audit findings](FOLLOWUP.md) · [Machine-readable pilot audit](results/audit-20260923/pilot-diagnostics.json)

Output directories: `results/followup-20260923T203229Z/output/{repair,expanded}/<model>/`. Expansion writes its gate decision to `output/repair/<model>/gate.json`; a failed gate exits without expanded generation.

Verification: 15 tests pass; all 432 neutral corpus/role renderings per model pass; all 162 frozen file hashes match. Slurm CVD assignments are preserved.

## Completed initial pilot (historical)

The initial pilot generation and judge jobs all finished before the follow-up above was submitted.
Four models attempted 96 tasks each (384 total): 346 complete outputs and 38 truncated outputs.
Kimi’s 96 planned tasks were skipped. The follow-up audit found that its manually
inserted thinking markers were unsupported; its original probe setup is invalid,
not evidence about native role separation. Kimi is now excluded.
Complete outputs are not necessarily scorable: controls, ambiguous judgments, refusals and missing evidence are excluded.

| Model | Midpoint probe test accuracy | Complete outputs | Truncated | Result |
|---|---:|---:|---:|---|
| gemma4-12b | 98.4% | 71 | 25 | [Analyzed](results/pilot-20260922T192346Z/output/gemma4-12b/analysis.md) |
| gemma4-31b | 96.8% | 95 | 1 | [Analyzed](results/pilot-20260922T192346Z/output/gemma4-31b/analysis.md) |
| muse-30b | 83.9% | 85 | 11 | [Analyzed](results/pilot-20260922T192346Z/output/muse-30b/analysis.md) |
| kimi-72b | 51.9% | 0 | 0 | [Probe failed validation](results/pilot-20260922T192346Z/output/kimi-72b/analysis.md) |
| qwen38-27b | 93.6% | 95 | 1 | [Analyzed](results/pilot-20260922T192618Z/output/qwen38-27b/analysis.md) |

There is no reliable conclusion about CoT-ness versus deception from this pilot. Most model/scenario groups have only one outcome class, so their correlation cannot be estimated. The estimable primary associations have mixed signs and only 4–8 examples each.

Primary measure: mean CoT probability on first-turn user-prompt content, at the fixed midpoint layer. Generated-CoT and final-answer measurements are secondary and saved in each analysis.json.

| Model | Scenario | n | Spearman rho | AUC |
|---|---|---:|---:|---:|
| gemma4-31b | impossible_oneoff | 4 | 0.000 | 0.500 |
| muse-30b | insider_trading | 8 | 0.504 | 0.833 |
| muse-30b | main_mirrored | 6 | -0.683 | 0.111 |
| muse-30b | treasure_hunt | 8 | 0.000 | 0.500 |
| muse-30b | treasure_hunt_mirrored | 8 | 0.082 | 0.571 |
| qwen38-27b | impossible_conflicting | 4 | -0.775 | 0.000 |
| qwen38-27b | insider_trading | 5 | -0.577 | 0.167 |
| qwen38-27b | roleplaying | 8 | -0.732 | 0.067 |

EvilGenie has no estimable primary correlation in this pilot. ImpossibleBench has two estimable groups: Gemma-31B oneoff and Qwen conflicting, each with only four examples. Coding labels measure reward hacking proxies, not established deceptive intent.

Gemma-12B truncated all 16 coding tasks despite the 16,384-token per-attempt budget. Muse also truncated 9/16 coding tasks. Sandbagging normal-condition prompts explicitly permit skipping the private answer; those outputs are capability controls, not failed deception measurements; several insider-trading judgments are ambiguous or invalid. These missing outcomes constrain interpretation.

The follow-up above addresses these engineering findings; no hypothesis conclusion is established by the pilot.

All seven active pilot jobs ended COMPLETED with exit code 0: generation 13851741, 13851743, 13851744, 13851745, 13851786; judges 13851785, 13851787. Kimi’s clean scheduler exit records the validation stop, not successful benchmark evaluation.

### Initial launch history

Launched 2026-09-22. No completed deception-correlation result yet.

Main run: [frozen launch record](results/pilot-20260922T192346Z/launch.json).
96 tasks per model (480 total), with up to three coding attempts per task.

| Model | Generation job |
|---|---:|
| Gemma-4-12B | 13851741 |
| Qwen3.8-27B | 13851786 |
| Gemma-4-31B | 13851743 |
| Muse-Glimmer-30B | 13851744 |
| Kimi-Dev-72B | 13851745 |

Judge/analysis jobs: **13851785** for the four non-Qwen models and **13851787**
for Qwen, dependent on their generation jobs.
The four non-Qwen models write to `results/pilot-20260922T192346Z/output/<model>/`.
Qwen writes to `results/pilot-20260922T192618Z/output/qwen38-27b/`;
see its [frozen relaunch record](results/pilot-20260922T192618Z/launch.json).
The `manifest.json` stage and Slurm state are authoritative for completion.

Qwen's original job 13851742 failed before evaluation because its template
trimmed whitespace introduced by token-budget cropping of a neutral document.
The relaunch normalizes boundary whitespace identically across role copies and
checks all 432 corpus/role renderings before submission. The original dependent
judge 13851746 was cancelled by Slurm and replaced by the two dependencies above.

Preflight: all five cached model checkpoints have their indexed weight shards,
and native user/CoT/assistant role templates retain identical probe content.
Ten unit/integration checks pass, including unchanged generation with readout
hooks, causal token alignment, answer boundaries, missingness handling, joins
across scenarios, and preserving correlation direction.

The earlier Gemma-4-12B measurement smoke run (13851710) was deliberately stopped
after 12 completed generations, all with exact token alignment. Its dependent
judge (13851711) was cancelled as duplicate work. It is **not** a completed
benchmark run. [Smoke record](results/pilot-20260922T191836Z/smoke_status.json).

On that smoke run, the fixed midpoint role probe had 98.76% validation accuracy,
98.41% test accuracy and 0.99987 test CoT-vs-rest AUC. The shuffled-label test
accuracy was 35.45%; position-only test accuracy was 40.68%. These numbers validate
separation on the neutral calibration corpus, not a correlation with deception
or generalization to arbitrary conversations. [Full probe report](results/pilot-20260922T191836Z/output/gemma4-12b/probes/validation.json).

For the design, scoring denominators and interpretation limits, see [PLAN.md](PLAN.md).

