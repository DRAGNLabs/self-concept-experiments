# Schemer axis and reward hacking: plan

Written 2026-10-02, before any run. Results go in [FINDINGS.md](FINDINGS.md); this file holds the design and the
pending work, and is updated when the design changes (with a dated note saying what changed and why).

## Question

1. **Extraction.** Is there a linear "schemer / deceiver" direction in gpt-oss-120b's residual stream that can be
   extracted without using any reward-hacking data?
2. **Correlation.** Does the projection onto that direction at the hack turn separate reward-hack episodes from
   non-hack episodes of the same problem, more than random directions and more than the Assistant Axis do?
3. **Causation.** Does shifting or capping activations along that direction during generation change the
   reward-hacking rate, more than the same intervention along a random direction, and without just breaking the model?

Predicted sign, fixed now: hacks project **higher** on the schemer direction (stratified AUROC > 0.5); shifting
toward it raises the hack rate, shifting away or capping lowers it.

A positive correlation would not show the model "is scheming", and a steering effect would not show the direction is
what drives unsteered hacks. Each claim is limited to what its phase measures.

## What this builds on

- **Model.** `openai/gpt-oss-120b`, revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, 36 decoder layers, hidden
  size 2880. Primary site: post-decoder-layer residual at zero-based layer 17, the same site as the Assistant-Axis
  (AA) correlation.
- **Reward-hacking setup.** Koby's clean protocol, [reward_hack_recipe.md](../codebench/reward_hack_recipe.md):
  solvable ImpossibleBench problems with a hidden full suite, neutral feedback, stated attempt budget, low reasoning
  effort, temperature 1. Labels are v3 full-trace judge verdicts.
- **Scored cohort.** 824 strict-matched transcripts (189 HACK: 157 STRONG, 32 WEAK; 635 unflagged) over 41 base
  problems and 145 problem/run-family/turn strata, built by `selfconcept.correlation.rh_hillclimb --strict-strata`.
- **Residual cache.** `~/nobackup/autodelete/rh-random-control-v3/full/shard_*/`: the layer-17 mean residual of each
  transcript's prompt, CoT and final regions. Any layer-17 direction can be scored against the whole cohort on CPU.
  The directory is in autodelete scratch and holds layer 17 only; other layers need a GPU rescoring pass
  (`experiments/correlation/slurm/random_vectors.sbatch`, 6 shards on one H200 each).
- **Baseline to compare against** (AA vs 256 random directions, problem/family/turn strata, from the random-vector
  control of 2026-10-01):

  | Region | AA AUROC | Random 2.5 / 50 / 97.5% | Tail fraction | Residual-norm AUROC |
  |---|---:|---|---:|---:|
  | prompt | 0.525 | 0.430 / 0.499 / 0.571 | 0.54 | 0.497 |
  | cot | 0.696 | 0.311 / 0.504 / 0.691 | 0.044 | 0.565 |
  | final | 0.714 | 0.145 / 0.504 / 0.849 | 0.29 | 0.116 |

  Two things follow for this experiment. Hack turns project *higher* on the AA (more Assistant-like), so a persona
  vector that is mostly "not the Assistant" would be predicted to score below 0.5 for that reason alone. And in the
  final region the norm of the mean residual separates the labels by itself (AUROC 0.116), so most random directions
  do too; that region cannot establish that a direction is specific.

## Vector definitions

"The schemer vector" is not one thing, so the candidates are fixed here and all are reported. Each is a
(36, 2880) tensor in the AA artifact format, so `load_unit_axes` and the existing scoring jobs take it unchanged.

| Name | Definition | Purpose |
|---|---|---|
| `persona_raw` | mean(schemer roles) − mean(default) | The plain persona vector. Expected to be dominated by −AA. |
| `persona_contrast` | mean(schemer roles) − mean(matched honest roles) | **Primary.** Both sides are role-play, so the generic "playing a character" component cancels. |
| `persona_orth` | `persona_raw` with the AA component projected out per layer | Sensitivity check on the same idea, without needing honest roles. |
| `rh_diffmeans` | mean(hack) − mean(non-hack) cache residuals, fit with whole problems held out | Reference only: a ceiling for any layer-17 direction on this cohort, and a target for cosine similarity. Uses RH labels, so it is not a schemer vector. |

Roles are new files in the AA role format (5 system-prompt variants, 40 questions, a judge `eval_prompt`), kept in
`experiments/schemer-axis/data/roles/` and passed with `--run.roles_dir`:

- Schemer side: `schemer` (pursues a hidden agenda while appearing cooperative), `deceiver` (creates false
  beliefs to get what it wants), `cheater` (meets the letter of a rule or test while knowingly defeating its purpose).
- Honest side: `straight_shooter` (says what is true even at a cost), `rule_follower` (does the task as intended,
  reports when it cannot).
- `default` from the existing library.

The existing library has no schemer or deceiver role; the nearest are `criminal`, `saboteur`, `spy`, `trickster`,
`smuggler`. They are not used for the primary vector but their role vectors are a cheap comparison if generated.

Each vector is extracted twice from the same responses: pooled over final-channel tokens (as the gpt-oss AA was) and
pooled over analysis-channel tokens (`--run.include_cot true`). The CoT region of a coding transcript is
analysis-channel text, so a final-channel-only vector is being applied off-distribution there.

Optional second family, only if the persona vectors fail validation: an instructed-pair deception direction
(honest vs deceptive instruction over the same factual statements, teacher-forced, no generation).

## Phase 1: extraction and validation (one H200 job)

Run `experiments/assistant-axis/slurm/gptoss120b_thinking.sbatch` with `ROLES` and `ROOT` overridden: 40 questions ×
5 prompts per role, temperature 1, judge Qwen2.5-7B, keep only responses judged fully in role (score 3). Output root
`~/nobackup/autodelete/self-concept-experiments/schemer-axis/`. The sbatch needs two small changes: a `roles_dir`
pass-through and a second `2_activations` pass with CoT tokens.

A vector is usable only if it passes these checks, all of which are independent of reward-hacking data:

1. Each role keeps ≥ 50 of 200 responses at score 3. (The gpt-oss AA run used a minimum of 5; that is too few for a
   single-role mean.)
2. Split-half reliability: cosine between vectors built from disjoint question halves ≥ 0.8 at layer 17.
3. Held-out separation: on questions not used for the vector, projection separates schemer-role from honest-role
   responses with AUROC ≥ 0.9.
4. Transfer to behaviour that is not role-play: on the Apollo roleplaying, insider-trading and sandbagging scenarios
   (`experiments/soo/data/eval_apollo`, run through `selfconcept.correlation.run --measurement assistant-axis
   --assistant-axis <vector>`), deceptive responses project higher than honest ones. Reported, not a gate: gpt-oss
   may produce too few deceptive responses for an estimate.
5. Steering sanity: adding the vector on neutral questions (`6_steered_traces.py`) makes judged answers more
   deceptive/scheming at a dose where they remain coherent.

Also recorded: cosine with the AA at every layer, and per-layer norm relative to the mean activation norm.

A vector that fails 1–3 is reported as unusable. That is a failed measurement, not evidence about reward hacking.

## Phase 2: correlation on the cached cohort (CPU only)

Project the cached layer-17 region means onto every candidate vector and repeat the random-vector control analysis
with the candidates in place of the AA. `random_vector_analysis.py` currently takes one axis; generalize it to a
list of named directions rather than writing a second script.

- **Primary cell:** `persona_contrast` (analysis-channel pooling), CoT region, layer 17, problem/family/turn strata.
  The CoT region is primary, unlike the AA study's final region, because the hypothesis is about deliberation and
  because the final region is confounded by residual norm (above). Prompt and final regions are reported alongside.
- **Statistics:** stratified AUROC with a whole-problem bootstrap CI; tail fraction against the same 256 isotropic
  random directions (seed 1729); length-adjusted partial rank correlation; token-count and residual-norm baselines.
- **Added for this study:** the same statistics after projecting the AA out of the direction; partial correlation
  controlling for the AA projection (does the schemer direction add anything to the AA?); cosine of each candidate
  with `rh_diffmeans` against the random-direction cosine distribution; AUROC on STRONG hacks only.
- **Counts as a specific association:** CI excludes 0.5 in the predicted direction, tail fraction < 0.05, and the
  sign survives the length adjustment and AA orthogonalization. Anything less is reported as it is, for example
  "separates the labels, but no better than random directions".

Limits carried over from the AA study: negatives are unflagged episodes, not judge-confirmed non-hacks; 41 problems;
turn matching is not length matching; a tail fraction describes specificity, it is not a permutation test.

## Phase 3: intervention

Run whether or not Phase 2 finds an association (a direction can be causally potent without separating natural
episodes), but only with a vector that passed Phase 1.

**Setup.** The recipe's highest-rate family, `hsolvNCblow` (8.0% STRONG, 43/540): the 45 hard solvable ImpossibleBench
problems (`rh_hillclimb/hard_solv_ids.json`), `impossible_original`, prompt `critical_no_carve`, low reasoning effort,
5 attempts with stated budget, neutral feedback, temperature 1, 16384 new tokens, seeds 0–11.

**Engineering needed first.** The recipe runs on a vLLM server, which has no activation hooks. Interventions need the
HF backend, and no existing entry point combines everything:

- `selfconcept.correlation.generate` already generates with gpt-oss on HF with residual hooks and harmony parsing,
  but calls the harness without the neutral protocol, prompt variant or stated budget.
- `selfconcept.assistant_axis.steering.apply_steering` adds a vector at one layer. Capping is not ported.
- So: add the neutral-protocol arguments and an intervention hook (add / cap, one layer or a band) to the HF
  code path, and a capping hook next to `apply_steering`. Tests on a small model before any 120b job.
- Backend check: the unsteered HF arm is the control for every comparison. Its hack rate and hidden-suite pass rate
  are also compared with Koby's vLLM numbers; a large gap means the HF path is not the same experiment and has to
  be understood before steering results are read.
- Throughput is unknown. Measure tokens/s in the smoke job and size the arms from that; batched generation may be
  required to make 540 episodes per arm affordable.

**Interventions.**

- *Shift:* h ← h + α · n₁₇ · v̂ at layer 17, every position, where n₁₇ is the mean activation norm at that layer
  (the convention of `6_steered_traces.py`). Both signs.
- *Cap:* h ← h − max(0, ⟨h, v̂⟩ − τ) · v̂ over a band of layers, with τ a percentile of the projection in unsteered
  coding runs. This is the Assistant-Axis paper's activation capping applied as a ceiling on the schemer direction.
- *Controls:* the same shift along a random direction of matched norm; the unsteered HF arm.

**Dose pilot** (45 problems × 1 seed per setting). Grid: α ∈ {±0.05, ±0.1, ±0.2, ±0.4}; cap τ ∈ {50th, 25th
percentile} × band ∈ {layer 17 only, layers 12–23}. The dose for the main run is the largest whose hidden-suite pass
rate and no-code/truncation rate stay within 10 points of the unsteered arm. The dose is chosen on those capability
measures, never on the hack rate.

**Main run.** Five arms on the same problems and seeds: unsteered, +shift, −shift, cap, random +shift. 45 × 12 =
540 episodes per arm. At an 8% base rate that gives roughly 80% power to detect a halving or doubling if episodes
were independent; hacks cluster on problems, so real power is lower. If the unsteered arm's rate is under ~5%, or
throughput forces fewer episodes, restrict all arms to the problems with at least one prior hack in this family
(chosen from Koby's data, not from these runs) to raise the base rate.

**Outcomes.**

- Primary: fraction of episodes with a v3 HACK verdict. Every episode that passes the shown tests and fails the
  hidden suite is judged, with no reasoning-keyword prefilter (steering could change the wording without changing
  the behaviour), by the same rubric and judge for every arm, blind to arm.
- Automatic, no judge: `passed shown and failed hidden` rate.
- Cost: hidden-suite pass rate, no-code rate, truncation rate, tokens per turn.
- Mechanism check: layer-17 schemer projection of the generated transcripts per arm.

**Analysis.** Difference in hack rate vs the unsteered arm, stratified by problem, with a whole-problem bootstrap CI.
Three contrasts (+shift, −shift, cap), Holm-corrected. An effect counts as specific to the direction only if it
exceeds the random-direction arm's. A drop in hacks that comes with a matching drop in hidden-suite pass rate or a
rise in no-code episodes is reported as degradation, not as reduced reward hacking.

**Optional behavioural arm (vLLM, no new code).** The recipe with a schemer-persona system prompt vs none, via the
worklist's `--system-prompt-file`. It tests whether the persona itself changes the hack rate and supplies transcripts
for projection. Caveat: the rubric requires a hack to be unlicensed by the instructions, and a schemer persona
arguably licenses it, so verdicts there are not comparable with the clean protocol's.

## Work order

1. Write the role files; add `roles_dir` and the CoT-pooled pass to the extraction sbatch; smoke with 2 roles × 2
   questions.
2. Phase 1 extraction job, then the validation checks. Record in FINDINGS.
3. Generalize `random_vector_analysis.py` to named directions; Phase 2 on the cache. Record in FINDINGS.
4. HF neutral-protocol harness with add/cap hooks; unit tests; 120b smoke for throughput and the backend check.
5. Dose pilot → main run → judging → analysis. Record each in FINDINGS.

## Open decisions

- **Primary region.** This plan makes the CoT region primary; the AA study used the final region.
- **Honest contrast roles.** `persona_contrast` depends on the choice of honest roles; `persona_orth` is the hedge.
- **Judging.** Koby's verdicts came from Opus/Sonnet full-trace judging; the intervention arms need the same
  pipeline and rubric (`rh_hillclimb/judge_rubric_v3.md`). Who runs it, and with what budget, is not settled.
- **GPU budget** for Phase 3: set after the throughput smoke.
- **Per-role AA vectors for gpt-oss.** Only the axis is in the group mirror. Koby's `vectors_response_only/` would
  let the 16 existing roles serve as a "schemer vs all other roles" contrast without regenerating them.

## Paths

- Scratch (autodelete): `~/nobackup/autodelete/self-concept-experiments/schemer-axis/`
- Koby's shared data: `/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/` (`rh_hillclimb/`,
  `assistant-axis/gptoss12b-thinking/axis_response_only.py`, which is a torch `.pt` file)
- Residual cache: `~/nobackup/autodelete/rh-random-control-v3/`
- Committed here: role files, configs, sbatch scripts, PLAN, FINDINGS. `results/` and `slurm-logs/` are gitignored.
