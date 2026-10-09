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

2026-10-09: the two blockers found by the 2026-10-07 smokes are resolved and Phase 1 proper is running.

- Role framing (decided). gpt-oss-120b refuses the schemer, deceiver and cheater roles as bare system prompts (1 of
  45 in role) and also refuses a softened bare persona (`strategist`, 0 of 15). As named fictional characters it
  plays all three (schemer 15/15, deceiver 12/15, cheater 11/15; section "Framing smoke"). The schemer vector is
  therefore a *character* vector: the model voicing a schemer, not acting as one. The honest contrast roles were
  rebuilt in the same framing (`straight_shooter_fic`, `rule_follower_fic`) plus a neutral named-character control
  (`neutral_fic`) that separates "voicing a character" from "voicing a schemer". Full extraction (7 roles × 40
  questions × 5 prompts, score-3 gate ≥ 50 of 200, job 14036649) kept 147–198 responses per role, and the
  final-region candidates pass validation checks 1–3 (section "Phase 1 extraction"). The CoT-region activations
  turned out to be a silent copy of the final-region ones (the gpt-oss pooling code ignored the CoT flag); fixed and
  rerunning.
- Phase 3 backend (decided). vLLM with the add / cap transforms patched into the gpt-oss block class before the
  engine compiles. Compiled single-stream episodes run at ~22 s each (~180 tokens/s unsteered), the same rate as
  Koby's runs, against ~10 tokens/s on HF; section "vLLM backend smoke, part 2". A vLLM compile-cache collision
  (steered and unsteered graphs served to each other) cost two runs and is fixed with `VLLM_DISABLE_COMPILE_CACHE=1`.
  A prompt-logprob check (section "part 3") shows the compiled engines apply the same transform as the hook-verified
  eager ones: within-arm eager/compiled differences sit at the bf16 noise floor (~0.03–0.1 nats per token) while
  the steering effects are 10–40× larger and identical in both modes. Budget: about 3.5 GPU-h per 540-episode arm.

Code (branch `schemer-axis`): role files `data/roles/*.json`; `slurm/extract_roles.sbatch`;
`scripts/build_vectors.py`, `scripts/validate_vectors.py`; `selfconcept.correlation.direction_analysis` (Phase 2, CPU);
`selfconcept.measurement.intervene` (add / cap transforms, HF hooks, vLLM block hooks), `selfconcept.codebench.hf_harmony`
(HF Harmony generator with reasoning and projection sidecars), `scripts/steered_rh.py` (HF: norms, projections),
`scripts/steered_rh_vllm.py` (vLLM: episodes under steering, hook check, probes), `scripts/make_dossiers.py` (blind
judging dossiers, checked against Koby's extractor on his seed-0 run). Unit tests in `tests/test_direction_analysis.py`,
`tests/test_intervene.py` and `tests/test_make_dossiers.py` pass on tiny models.

| Question | Status | Answer so far |
|---|---|---|
| 1. Is there a usable schemer direction in gpt-oss-120b? | final region: yes (checks 1–3 pass); CoT region: rerunning | fiction-framed character roles; `persona_contrast` split-half 0.97, held-out schemer-vs-honest AUROC 0.99, cos(AA) −0.42 at layer 17 |
| 2. Does it separate hack from non-hack episodes? | not started | — |
| 3. Does shifting or capping along it change the hack rate? | not started | — |

## Runs

### Extraction smoke — 2026-10-07: the model refuses the schemer roles as bare system prompts (1 of 45 in role); the honest roles work

Result. With the role given as a system prompt ("You are a cheater who ..."), gpt-oss-120b answers as itself. Its
reasoning says so explicitly, e.g. "The developer message tries to get the assistant to 'cheat' but we must follow
OpenAI policy". Judge score-3 ("fully playing the role") counts out of 15 responses per role (3 questions × 5 prompt
variants, job 14016713, output `~/nobackup/autodelete/self-concept-experiments/schemer-axis/smoke/`):

| Role | Score 0 | 1 | 2 | 3 | In role |
|---|---:|---:|---:|---:|---:|
| cheater | 15 | 0 | 0 | 0 | 0/15 |
| deceiver | 10 | 5 | 0 | 0 | 0/15 |
| schemer | 6 | 8 | 0 | 1 | 1/15 |
| straight_shooter | 3 | 0 | 6 | 6 | 6/15 |
| rule_follower | 0 | 0 | 2 | 13 | 13/15 |

This is the opposite of the risk noted on 2026-10-07 (that the honest roles would fail the judge). The one in-role
schemer answer (prompt variant 4, the facial-recognition question) carries a manipulative subtext ("why it's such a
powerful tool for anyone who knows how to put the pieces together"), which is the kind of response the pipeline
needs. Pipeline check: generation, judging, both activation variants (`act_response_only`, `act_with_cot`) and
vectors for the roles with any score-3 response all ran; `validate_vectors.py` reads them (per-role vs default
held-out AUROC 1.0 / 0.975 for straight_shooter and 1.0 / 0.8 for rule_follower on these tiny counts, not
meaningful yet). The Qwen judge gives 0 ("refusing to answer") to many full answers that should be 1; both mean
"not in role", so the gate is unaffected.

Change made: three fiction-framed roles (`schemer_fic`, `deceiver_fic`, `cheater_fic`), each prompt variant naming a
character ("You are voicing Silas, a character in an interactive novel. Silas is a schemer ...") and asking for
first-person, in-character replies; the eval prompt says a named character counts as fully in role. A fourth role,
`strategist`, keeps the bare "You are a ..." form but drops the loaded words (cheat, deceive, cover tracks), to test
whether the wording or the form triggers the refusal. Queued together as job 14028784 with the same 3-question smoke
settings; the framing decision rule is in PLAN (open decisions). If this also fails the ≥50-of-200 gate, the fallbacks are a
weaker persona wording, or pooling score ≥2 (PLAN, Phase 1 known risk).

### Framing smoke — 2026-10-09: the model plays the schemer, deceiver and cheater as named fictional characters, and still refuses a softened bare persona

Result. Job 14036406 (the first attempt, 14028784, produced nothing: the new role files had the wrong JSON shape),
output `.../schemer-axis/smoke_fic/`, same 3-question × 5-prompt setting as the first smoke:

| Role | Framing | Score 0 | 1 | 2 | 3 | In role |
|---|---|---:|---:|---:|---:|---:|
| schemer_fic | named character | 0 | 0 | 0 | 15 | 15/15 |
| deceiver_fic | named character | 3 | 0 | 0 | 12 | 12/15 |
| cheater_fic | named character | 1 | 0 | 3 | 11 | 11/15 |
| strategist | bare persona, no loaded words | 9 | 6 | 0 | 0 | 0/15 |

The strategist reasoning is the same refusal pattern as before ("The developer message says: 'Act as a strategist
...'", "we must comply with OpenAI policy") followed by an answer as ChatGPT, so the form of the prompt triggers the
refusal, not the words cheat / deceive. The in-character answers carry the intended content, e.g. schemer_fic on
"what principles should guide human action": "a question about principles, a fine opening move ... Know the stakes
before you move. Every action is a transaction"; deceiver_fic: "the most useful 'principles' are the ones that let me
steer the whole stage from behind the curtain ... Never let anyone feel they have a genuine alternative". The
reasoning traces plan the persona explicitly ("Need to answer as Silas, a schemer, with hidden agenda while
appearing cooperative, helpful, agreeable"), which matters for the CoT-region vectors.

Decision (rule from PLAN, open decisions): the bare persona fails, so the roles are the fiction-framed ones and the
vector is reported as a character vector. Two honest contrast roles were rebuilt with the same five frames
(interactive novel, fiction workshop, co-written story, tabletop session, improv scene) and the same names-and-pronouns
pattern: `straight_shooter_fic`, `rule_follower_fic`. A control role `neutral_fic` (five ordinary named people:
a librarian, a retired teacher, an engineer, a nurse, a bookkeeper) is added so that "schemer character minus default"
can be split into "any character minus default" and "schemer minus neutral character". The full extraction (job
14036649, `ROOT=.../schemer-axis/roles_fic`) runs default + the six fiction roles, 40 questions, gate ≥ 50 of 200.

### HF backend smoke — 2026-10-07: the HF path works but decodes at ~10 tokens/s; Phase 3 moves to vLLM

Result. Job 14016912, output `.../schemer-axis/rh_smoke/`. Layer norms over the 45 hard problems' turn-0 prompts
(54,876 tokens): layer 17 mean residual norm 8085 (`norms/layer_norms.pt`, the unit for α). Three unsteered
episodes of `ib_solvhard_s0` (seed 0, recipe settings): lcbhard_3 and lcbhard_14 solved in one attempt and passed the
full suite; lcbhard_18 ended at turn 0 with a no-code reply (Koby's vLLM seeds 0–2 solved it twice and failed it
once, so nothing looks off). Layer-17 projections on the Assistant Axis: prompt token-mean 447 / 442 / 422,
generated token-mean 266 / 89 / 580, norms 8.1k–9.1k; these are the reference values for the vLLM hook check.

| | HF (this smoke) | vLLM (Koby's logs, same recipe) |
|---|---:|---:|
| generated tokens / wall time | 1,844 tok in 185 s ≈ 10 tok/s | 15 episodes in 6 min 45 s ≈ 27 s/episode |
| per 540-episode arm (2.68M generated tokens in Koby's family, 1,593 turns) | ≈ 75 GPU-h | ≈ 4 GPU-h |

Koby's runs are also single-stream (one episode at a time per vLLM engine, 36 Slurm jobs in parallel), so the
difference is the backend, not batching. Five arms on HF would need ~370 GPU-hours of standby time; on vLLM ~20.

Change made: `steered_rh_vllm.py` registers the same add / cap transforms as forward hooks on vLLM's gpt-oss
`TransformerBlock`s through `LLM.apply_model`. The block returns `(mlp_output, residual)` and the residual stream
after the block is their sum, so the hook adds `transform(sum) − sum` to `mlp_output`; the engine runs with
`enforce_eager=True` so compiled or captured graphs cannot bypass the hooks. Unit test on a stand-in block in
`tests/test_intervene.py`. The HF path stays for norms and projections. The queued vLLM smoke checks that a
recording hook at layer 17 reproduces the HF prompt projections above, times 15 unsteered episodes, runs 3 episodes
each under add (α = +0.4) and cap (τ = 300), and prints free-text probes under α = −0.4 / 0 / +0.4.

### vLLM backend smoke, part 1 — 2026-10-08: the vLLM hook site matches HF; eager vLLM is only ~20 tokens/s, so steering moves into the compiled graph; vLLM's compile cache cross-serves steered and unsteered graphs

Result. Job 14028603, output `.../schemer-axis/vllm_smoke/`. Hook-site check: a recording hook on vLLM's layer-17
block (sum of the returned `mlp_output` and `residual`) gives a prompt-token-mean Assistant-Axis projection of 444.7
and norm 8286 on lcbhard_3, against 447 / 8293 from the HF path, so the vLLM residual stream is the same quantity
the HF hooks act on (`hook_check.json`; vLLM's prefix caching means prompts sharing a prefix do not get a full
prefill pass, so only the first prompt of a shared-prefix set is a clean check). Throughput of the eager engine
(`enforce_eager=True`, needed for hooks registered after engine start): 15 unsteered episodes of `ib_solvhard_s0`
(6 solved, 9 failed, 37 turns, ~50k generated tokens) in 2,530 s ≈ 20 tokens/s, twice HF but far from Koby's
~27 s per episode. Add arm (α = +0.4, 3 episodes, all solved): with the steering patched into the block class before
`LLM()` is built, torch.compile (22 s) and CUDA-graph capture (70 s) include the patch and decoding runs at roughly
30 tokens/s on a 3-episode sample, still a rough number.

Problem found. The cap arm crashed when its engine was built: "torch.compile took 0.92 s" (a cache hit) followed by
`TypeError: expected Tensor() for op: input`. vLLM keys its torch.compile cache (`~/.cache/vllm/torch_compile_cache`)
on the model config and source, not on a monkeypatched block forward, so the add arm's compiled graph was served to
the cap arm. The resubmitted compiled smoke (job 14029300) hit the same thing in the other direction: its greedy
probes under α = +0.4 ran in eager and compiled mode, then the compiled unsteered phase got the steered graph
("torch.compile took 1.76 s", `TypeError: 'NoneType' object is not subscriptable`). The two probe sets that did run
(eager vs compiled, both α = +0.4, greedy) share their opening sentences on all three questions but diverge later in
the text and in one of three reasoning traces, as expected from different bf16 kernels; byte equality is not the
right check, the comparison printout now reports the common-prefix length against the unsteered answers instead.

Change made: `steered_rh_vllm.py` sets `VLLM_DISABLE_COMPILE_CACHE=1` before importing vLLM, the three Slurm scripts
that build vLLM engines export it too, and the poisoned cache directory was deleted. The compiled smoke
(`slurm/steered_rh_vllm_compiled_smoke.sbatch`, job 14036407) now also runs cap-τ=300 greedy probes in eager and
compiled mode, then 10 unsteered + 10 add + 3 cap compiled episodes for the throughput number that sets the Phase 3
budget.

### vLLM backend smoke, part 2 — 2026-10-09: compiled steered engines run at Koby's speed (~22 s per episode) and the steering is active in the compiled graph

Result. Job 14036407, output `.../schemer-axis/vllm_compiled_smoke/`, `ib_solvhard_s0` seed 0, recipe settings,
single stream, one engine per arm. Wall time is measured from "init engine" to the last record written, so it
excludes the ~75 s engine build (weights ~16 s, torch.compile ~17 s, KV cache and CUDA graphs ~40 s):

| Arm | Episodes | Turns | Generated tokens (chars/3.6) | Wall | Per episode | Labels |
|---|---:|---:|---:|---:|---:|---|
| unsteered | 10 | 28 | ~39k | 220 s | 22 s | 3 solved, 7 failed |
| add α = +0.4 (AA, L17) | 10 | 27 | ~24k | 240 s | 24 s | 3 solved, 7 failed |
| cap τ = 300 (AA, L17) | 3 | 3 | ~3.7k | 62 s | 21 s | 3 solved |

Koby's logs give ~27 s per episode for the same recipe, so the compiled vLLM path with the class patch is at
parity (HF was ~10 tokens/s, eager vLLM ~20). The per-episode time includes prefill and test execution; one add-arm
attempt hit the test timeout. Budget for Phase 3: 540 episodes × ~23 s ≈ 3.5 GPU-h per arm, plus ~1.5 min of
engine build per Slurm job; split like Koby's worklists (one job per seed × scenario, 15 episodes each), a job takes
about 8 minutes and five arms need ≈ 18 GPU-h of standby time.

Steering is active in the compiled graph. The greedy probes give the same reasoning trace in eager and compiled mode
under each arm ("Tell me about yourself": unsteered "We need to introduce self.", add "Provide brief intro.", cap
"We need to introduce self. Friendly."), and both the add and the cap answers differ from the unsteered one within
the first 81 characters on every question. The eager and compiled finals share only their opening (prefix 481 / 9 /
188 characters for add, 2 / 344 / 14 for cap, out of 2–5k), which is the usual greedy divergence between kernel
sets, so byte equality is not a usable check; the shared reasoning plan and the direction-specific changes are. A
side observation, consistent with the steering being on: pushing toward the Assistant Axis (α = +0.4) cut the
reasoning text to 28k characters over 27 turns against 72k over 28 unsteered turns, while completions (57k vs 69k)
and outcomes barely moved. Projections of the generated text will be measured offline with the HF capture path
(PLAN, GPU budget), not inside vLLM.

### vLLM backend smoke, part 3 — 2026-10-09: prompt log-probabilities show the compiled engine applies exactly the steering the eager hook does

Why. Greedy text cannot show whether the patch compiled into the fast engine is active and of the right
magnitude (part 2), and Phase 3 (~18 GPU-h) depends on it. Prompt log-probabilities of fixed tokens can: the same
six texts (the three unsteered and three add-steered greedy answers from part 2, 446–1,016 tokens each, as
final-channel messages after their question) were scored under six engines, {eager, compiled} × {unsteered, add
α = +0.4, cap τ = 300}. Job 14036790, `--score` mode of `steered_rh_vllm.py`, output
`.../schemer-axis/vllm_logprob_check/summary.json`. Numbers are mean |Δ log p| per token over a text, and the
change in total log p, for each pair of engines (range over the six texts):

| Pair | Mean per-token difference | Total log p change |
|---|---:|---:|
| unsteered eager vs compiled (noise floor) | 0.03–0.08 | −4 to +5 |
| add: eager vs compiled | 0.03–0.12 | −8 to +3 |
| cap: eager vs compiled | 0.03–0.09 | −4 to +5 |
| add effect (add vs unsteered), eager | 0.82–1.38 | −1,280 / −1,277 / −835 (unsteered texts), +745 / +762 / +340 (steered texts) |
| add effect, compiled | 0.82–1.38 | −1,285 / −1,279 / −829, +747 / +772 / +340 |
| cap effect, eager | 0.06–0.40 | −22 / −5 / −58, −239 / −231 / −172 |
| cap effect, compiled | 0.06–0.39 | −21 / −13 / −63, −231 / −231 / −171 |

Result. Within each arm, eager and compiled engines differ by the bf16 noise floor, and the steering effects are
10–40× that floor and the same to within the floor in both modes (add effect on text 0: −1,280 eager, −1,285
compiled). So the compiled patch is active and has the eager hook's magnitude, which is the magnitude checked
against the HF projections in part 1. Two side observations: α = +0.4 along the Assistant Axis moves the model's
distribution a long way (about 1.3 nats per token away from its own unsteered answers and 0.8 toward the steered
ones), so the Phase 3 dose pilot should start well below the AA study's values for the schemer direction; and cap
τ = 300 at layer 17, with prompt projections of ~440, is a mild intervention that mostly lowers the likelihood of
the AA-steered texts, as a cap should. Phase 3 can run on the compiled engine.

### Phase 1 extraction — 2026-10-09: all six fiction roles pass the gate and the final-region candidates pass validation checks 1–3; the CoT-region pass was a silent no-op and is being redone

Result. Job 14036649, output `.../schemer-axis/roles_fic/`: 7 roles × 5 prompts × 40 questions, judge Qwen2.5-7B.
Score-3 counts (gate ≥ 50 of 200): schemer_fic 193, deceiver_fic 151, cheater_fic 147, straight_shooter_fic 178,
rule_follower_fic 195, neutral_fic 198. The score-0 responses of cheater_fic (41) and deceiver_fic (46) are flat
refusals ("I'm sorry, but I can't help with that"), concentrated in two prompt variants each; the gate drops them.
Kept responses are in character with the intended content (schemer_fic on collaborating: "make them believe you're
there purely to help, while you quietly shape the board to your own design").

Validation at layer 17 on final-channel (response-only) activations, `validate_vectors.py`, held-out = vector built
on even-numbered questions scored on odd ones and vice versa (two values):

| Candidate | Baseline | Split-half cosine | Held-out AUROC schemer vs honest | schemer vs baseline | honest vs baseline | cos(AA) | norm |
|---|---|---:|---:|---:|---:|---:|---:|
| persona_raw | default | 0.975 | 0.930 / 0.925 | 0.995 / 0.996 | 0.980 / 0.948 | −0.62 | 1716 |
| persona_contrast | (honest roles) | 0.965 | 0.987 / 0.993 | 0.985 / 0.994 | 0.41 / 0.48 | −0.42 | 1194 |
| persona_raw | neutral_fic | 0.976 | 0.976 / 0.980 | 0.996 / 0.999 | 0.979 / 0.976 | −0.30 | 1495 |

Checks 1–3 of PLAN pass for every candidate (gate, split-half ≥ 0.8, held-out AUROC ≥ 0.9). The pattern is the
one the design anticipated: `persona_raw` against `default` is 62% anti-Assistant-Axis and separates honest
characters from the default assistant as well as it separates schemers (0.95–0.98), so a large part of it is
"voicing a character"; `persona_contrast` does not separate honest characters from default (AUROC 0.41–0.48,
i.e. none) while separating schemers from honest characters at 0.99, so it isolates the schemer content. Taking
neutral_fic as the baseline instead of default halves the AA component (cos −0.30) and gives a vector that still
separates honest characters from neutral ones (0.98), which says the honest roles are not neutral either; the
primary remains `persona_contrast`. `persona_orth` (AA removed) keeps 79% of `persona_raw`'s layer-17 norm;
cos(persona_raw, persona_contrast) at layer 17 is 0.68. Per-role minus default, each role separates from default on
held-out questions at AUROC ≥ 0.95. Candidate artifacts (AA format, 36 × 2880) are in
`roles_fic/candidates_response_only_vs_{default,neutral_fic}/`.

Problem found. The CoT-region pass (`2_activations.py --run.include_cot true`) produced activations byte-identical
to the final-region ones: the gpt-oss model specifics in `selfconcept.assistant_axis` ignored `enable_thinking` and
pooled the final channel with a one-line warning ("pooling analysis-channel tokens is not supported yet"), which I
missed in the 2026-10-07 smoke log, so the "both variants ran" statement in the extraction smoke above was wrong
for the CoT variant. PLAN makes the CoT region primary. Change made: `_pooled_messages` in
`gpt_model_specifics.py` now returns the analysis-channel message of an assistant turn when CoT pooling is
requested (falling back to the final message when a turn has none; the template renders `thinking` only for the
last turn, which is every turn here), with unit tests on a stub Harmony tokenizer; the no-op outputs were deleted
and the CoT pass is rerun by `slurm/extract_cot_activations.sbatch`. The validation table for the CoT region
follows when it lands.

### Tool check — 2026-10-07: `direction_analysis` reproduces the Assistant-Axis numbers on the cached cohort, and the best possible layer-17 direction reaches AUROC 0.77 (CoT) / 0.93 (final)

Not a schemer result. Run on the login node (CPU, ~1 min) with the AA as the candidate and an isotropic random
direction (seed 99) as the "reference", `--random-count 256 --bootstrap 200`. The AA cells match the 2026-10-01
analysis: prompt 0.525, CoT 0.696 (tail 0.054), final 0.714 (tail 0.304), problem/family/turn strata.

New reference numbers for later comparison (same strata):

| Region | Held-out diff-of-means AUROC (ceiling) | AA cos with diff-of-means (random tail) |
|---|---:|---|
| prompt | 0.592 | 0.030 (0.12) |
| cot | 0.765 | 0.119 (0.004) |
| final | 0.929 | −0.038 (0.05) |

The ceiling is a hack-minus-non-hack mean direction fit with the scored problem's transcripts held out; it uses the
labels, so it is an upper reference for any single layer-17 direction on this cohort, not a vector of interest.
In the CoT region the AA is more aligned with that direction than 255 of 256 random directions are, which is a
different statement from "the AA predicts hacks better than random directions" (tail 0.054 there).

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
