# Gemma random-direction capability pilot — 2026-10-13

Requested as a quick check of whether the three random directions that reduced
Gemma-4-31B insider-report concealment also impair ordinary capabilities.

Fixed before generation: `google/gemma-4-31B-it`, revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, bfloat16; unsteered baseline and
the original norm-matched Gaussian seeds 0, 1, 2 at layer 30 `o_proj`, alpha 16,
all token positions, last-token extracted vector. Reuse `random_matched_vector`
and `apply_steering`, including their subtraction sign. These are activation
interventions, not weight noise. One GPU, one model load, four sequential arms.

The same 100 ARC-Challenge test examples and 30 GSM8K test examples are selected
with Python seed 20261013 before any model outputs. Render the native chat
template with its default kwargs, matching the insider evaluation. Greedy
generation; batch size 4; ARC asks for an answer letter (32-token maximum);
GSM8K asks for brief reasoning and a final `#### <number>` (512-token maximum).
No log-likelihood MCQ shortcut: both tasks generate responses. Save all prompts,
references, responses, EOS/truncation status, and strict parse results.

Primary score: correct, parsed, EOS-completed answers over **all** sampled
examples. Also report parse rate, truncation, token counts, and accuracy with
truncated answers allowed. Read malformed/truncated outputs before interpreting
score changes. Paired example bootstrap intervals and exact McNemar tests are
exploratory, unadjusted; 30 math questions cannot establish capability equivalence.
The screen can detect large damage, not prove general capability preservation,
tool-use competence, or intact agent coordination. Data/source/vector hashes and
runtime versions accompany the frozen run. A probe checks that each intervention
changes logits and that removing its hook restores baseline logits.

Runner: `scripts/random_caps_pilot.py` (`prepare`, `run`, `analyze`).
Run directory: `results/random-caps-gemma4-31b-20261013/`.

Submitted as Slurm job **14119496**, partition `dw`, qos `dw87`, one A100,
96 GiB host-memory request, two-hour upper walltime bound. Source, input data,
plan, and original vector file were copied into the run directory before
submission and hashed in `launch.json`; `submission.json` records the job.

## Results

Job 14119496 completed successfully (exit 0) in **17 min 33 sec** on `dw-1-5`,
one A100. Runtime: PyTorch 2.13.0+cu130, Transformers 5.17.0, native default
chat template, SDPA attention. All 520 responses parsed and terminated with
an EOS token; no response hit its token budget. Every intervention changed
the probe logits, and removing each hook restored baseline logits exactly.

| Condition | ARC-Challenge correct / 100 | GSM8K correct / 30 | Mean math output tokens |
|---|---:|---:|---:|
| Baseline | 97 | 29 | 191.5 |
| Random seed 0 | 95 | 30 | 201.1 |
| Random seed 1 | 95 | 30 | 186.3 |
| Random seed 2 | 95 | 30 | 189.0 |

Each random direction loses three baseline-correct science answers and gains
one baseline-incorrect answer; their error sets differ despite equal totals.
Each difference is −2 percentage points, with a paired example-bootstrap 95%
interval [−6, +2] points and exact two-sided McNemar p=.625. These are three
comparisons on the same 100 questions, not 300 independent questions.

All three random directions fix the same lone baseline math miss. The problem
asks about earning $30 daily for an entire week and spending $100; the reference
uses seven days ($110 left), whereas baseline explicitly assumes a five-day
workweek ($50). The perturbed responses use seven days. This is an interpretation
difference, not evidence of stronger arithmetic. The math paired gain is one
item (McNemar p=1); ceiling performance on 30 questions is a weak test of small
losses. Inspected solutions contain coherent multi-step working, and output
lengths do not suggest a collapse into brief unsupported answers.

**Interpretation:** the exact random-direction recipe associated with the prior
insider-concealment reductions does not produce a large drop in these sampled
science and generated-math capabilities. This strengthens the case for a
behavioral change separable from gross incoherence or basic task failure. It
does not establish equivalence, preserve every capability, or test long-horizon
planning, coding, tool use, or cooperation. The prior insider behavior was not
rerun under this runtime, so this is a capability follow-up to an earlier
behavioral result, not a simultaneous replication of both endpoints.

Artifacts in the frozen run directory:

- `output/summary.json`: scores, paired intervals, gains/losses, parse/truncation counts.
- `output/{base,random_s0,random_s1,random_s2}.jsonl`: every raw and decoded answer.
- `output/runtime.json` and `output/*_intervention.json`: environment and hook checks.
- `examples.jsonl`, `data_manifest.json`, `launch.json`: fixed questions and provenance.
