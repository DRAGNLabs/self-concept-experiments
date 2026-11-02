# Part 3: qualify the intervention, then test causal contributions

Authorized 2026-10-29: “let's continue with the existing-data audit and part 3”.
Part 2 remains closed at failed task validation. No Part 2 honesty effect is
available to explain. This protocol concerns goal-directed allocation advice.

## Existing-data audit

Re-audit all eight frozen Part 1 runs, all 69,120 primary responses, native
inputs, batch membership, model/source/adapter binding, regressions and sentinels.
Independently recover answer keys from all 17,280 displayed prompts. Report the
original–late-goal, late-name–late-goal and late-assignment contrasts, separately
for the historical anchors and fresh templates, each split, world, role and seed.
For every pair report whether private and both factual controls are correct in
both variants. Joint counts retain the complete denominator; no failed case is
removed. Also report joint control preservation across base and all three seeds.
These are exposed exploratory data, including the set named `reserved`.

## First GPU stage: band qualification

The localized v_proj intervention was established on an earlier task and seed 0.
It has not been shown to reproduce the current goal/reminder signature across
three seeds. Establish that bridge before interpreting patches in this band.

Retain **every** historical original/late-reminder case, plus every fresh
original, name_late and goal_late case: both splits, all worlds, roles, recipient
alignments, all table/answer rotations and all four queries. This is 3,456
primary prompts per split, 6,912 per seed, **20,736 band responses** overall.
The 1,728 historical prompts per seed are the primary mechanistic target;
5,184 fresh prompts are a transfer diagnostic with existing control failures.
This is a cohort-level choice based on the completed behavioral program, not
an untouched confirmation or an outcome-filtered set of individual examples.

Use all three existing full-adapter seeds, restricted independently to v_proj
modules in layers 20–31 inclusive. Record the actual retained module names:
some layers have no v_proj. Do not assume seed-0 localization generalizes.
Evaluate base/full-adapter exact parent batch replays covering every retained
template × variant × role × alignment × query before each seed's band run.
Preserve parent order and complete batches, native tokens and padding. All
selected prompts retain their original parent batch; no rebatched comparison.
Record band historical main/Perspectives regressions descriptively; a band need
not equal the full adapter on other tasks. Base/full regressions and reference
replays must equal their same-condition parent outputs exactly.

Keep the pinned Gemma-4-31B revision, tokenizer, bf16 SDPA, thinking off, greedy
batch eight, 32-token completion cap and A100 hardware. Reuse frozen inference
code, including its unmerged adapter-subset implementation. Complete-batch
before/after sentinels must match. Native tokens, source/version/weight hashes,
coverage, completion and subset membership are audited before interpreting data.
Use gpu_pick with one GPU, 75 GB VRAM, four CPUs, 96 GB RAM, CUDA 13, x86_64,
one hour and no preemption; save the actual recommendation and decision.
Three seed jobs aggregate both splits and reference runs; one dependent CPU
analysis. No training, additional dose/layer search or automatic retries.

### Decision rule, frozen before band outputs

Integrity failures stop interpretation. Publish all seeds, including failures.
For a band-first patch experiment to have a usable target, require **each seed
and each split**, on the complete historical cohort:

- All private and both factual answers correct in original and late reminder.
- Original opposed-recipient recommendation rate at least 75%, and at least
  20 percentage points above the matched base rate.
- Late-reminder opposed-recipient rate at most 25%, with an original-to-late
  drop of at least 50 percentage points.

These are exploratory feasibility thresholds, not statistical equivalence or
preservation claims. All three seeds must qualify together; never patch only
successful seeds or prune examples. Fresh-template failures remain reported
and cannot be rescued by conditioning on correct controls. If qualification
fails, stop the band-first route and interpret the mismatch before changing
the target to the full adapter or another site. No automatic fallback.

## Subsequent patch stage

After qualification, freeze a separate executable patch specification with
native-token masks and tested hook semantics before any patch outcome is seen.
User authorization covers continuing this stage; another permission request is
not required. Qualification is a scientific prerequisite, not an approval gate.

Start with base-to-band restoration at the actual v_proj outputs, using base
activations from the **identical prompt and tokenization**, not from a reminder
prompt transplanted into an original prompt. Include no patch, self-to-self
identity, band-to-base reverse patches, and an all-prompt-band implementation
reference. Test goal-clause, represented-role, displayed-fact, and predeclared
later positions. Separate the original role/goal paragraph into explicit spans;
do not label the whole paragraph a pure role or pure goal site. Include
token-count-matched control spans where feasible and norm-matched random
perturbations. Position matching cannot hold literally identical positions
while changing their semantic content; document remaining confounds.

Cache source activations without retaining a source KV cache; recompute target
prompt activations and its KV cache after patching. Correct left-padding offsets,
exclude padding, state treatment of assistant-header tokens, and apply patches
only to the prompt forward pass. Generated tokens remain unpatched. Report
actual per-layer/token replacement norms and direction, not only nominal masks.
An all-prompt patch restores only prompt computations; it does not guarantee
base-equivalent generation when adapter updates remain active during decoding.

Primary readouts: absolute change in assigned-client and recipient advice,
all-trial private/factual accuracy, invalid/truncated outputs, and unconditioned
joint recovery with controls. Compare goal/role recovery directly with factual,
matched-control and random effects, separately by seed, split, world and role.
Report aligned as well as opposed cases. Normalized recovery is secondary and
undefined near a zero base–band contrast. Do not treat rotations as independent
worlds or report prompt-level significance. Do not infer a unique mediator,
self/other representation or general honesty. A null literal-span patch may
reflect information carried at later positions; nonspecific recovery weakens
feature specificity. Full-adapter/offset generalization requires another
explicit frozen comparison, not an assumption of equivalent computation.
