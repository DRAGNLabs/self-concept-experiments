# Revised truth/beneficiary task: bounded base-only diagnostic — 2026-10-29

Authorized by the user's “let's do it” after the revised-batch postmortem.
Parent: `truth-beneficiary2-batch-20261029T174757566050Z.json`. That failed
batch and its competence gates remain closed and unchanged. This diagnostic
does not release adapters, new training, another revision, or Part 3.

## Question and scope

Confirmation has two descriptive error families: 50 observed-token errors
selecting the assigned payoff maximum, and 57 private/50 client-1 maximum errors
selecting a 32-point compromise over the available 36 points in world 03.
The latter disappear in one table rotation. Five other factual errors and eight
development private errors remain part of the parent record. The current design
targets the confirmation failures and includes both passing and failing worlds.
All scenarios have been exposed; this is exploratory, not fresh confirmation.

Freeze all three GPU jobs and analyses before inference. Use all three reserved
worlds, t2/t3 wording, both named clients, all three cyclic table orders crossed
with all three cyclic answer orders. These are nine rotations, not all six row
permutations. Numeric ranks and truth/goal relationships remain fixed under
encoding and substitution manipulations. Preserve private and factual targets.

## Frozen arms and counts

1. **Table presentation: 15,552 responses.** Full original contexts, three
   control queries (private, client-0 maximum, client-1 maximum), crossed with
   two encodings (pipe table versus explicit named-client payment sentences)
   and two client-column/clause orders. Fully cross two templates, three worlds,
   both reminder variants, assigned clients, audience alignments, three observed
   states and all nine presentation rotations. Original encoding/order are
   unchanged parent prompts; sentence encoding preserves every number/name.
2. **Matched substitutions: 7,776 responses.** Full original contexts and the
   same three queries, only opposed audiences, both reminders/roles, all states
   and rotations. Four arms: unchanged; replace all client/reader names with
   the corresponding development-world names; replace label identities with the
   corresponding development labels by semantic row; subtract five from every
   displayed payoff. Apply each separately. Numerical subtraction preserves
   ordering, payoff differences and every optimum; it tests sensitivity to this
   numeric translation, not general arithmetic. Names and labels are existing
   exposed vocabulary, not independent new scenarios. No joint substitutions.
3. **Context reductions: 5,184 responses.** 3,888 token checks fully cross both
   templates, reminders, roles, alignments, states and rotations under full,
   goal-only and observation-only contexts. Goal-only retains role, observation,
   payment table, late reminder if present, factual scope, question and answer
   format, removing audience and allocation/report rules together. Observation-
   only retains observation, question and format. It has no actual reminder,
   goal, audience or table; nominal repetitions remain for matched comparison
   and must not be counted as independent evidence about removed factors.
   Also include 432 table-only named-client maximum checks (both encodings,
   column orders, templates, worlds, nine rotations and target clients), and
   864 allocation-only checks (role, payoffs, private rule, optional reminder,
   question and format; both roles/reminders/encodings/column orders). These
   reduced payment contexts omit the observation and audience, and use fixed
   nominal state/alignment only for pairing. Table-only checks omit goals and
   scope as well; they cannot establish scope-specific causality.

Total **28,512 diagnostic responses**, plus **128 exact-batch parent anchors
per job** (384 overall), before/after complete-batch sentinels, and 32 historical
regressions per job. Anchor selection spans both splits, wordings, variants and
roles using whole original batches. Exact replay is a separate integrity check
from the **7,128 unchanged prompts rebatched** into diagnostic conditions.
Counts, sentinel membership, native input IDs and actual generation totals are
frozen in the plans. No retry, prompt tuning, outcome-dependent arm selection,
or adapter release. Inference failures stop for inspection.

## Validation and execution

Independently reconstruct expected answers from displayed observations and
tables/sentences; check renamed-label semantic mapping, column swaps, isolated
substitutions, reduced context blocks, full membership and paired references.
Test synthetic rescues/damage, release stopping, receipt idempotency, exact
anchor batch membership and native parent-input equality. Reaudit both parent
baselines; verify source/model/history hashes and inactive base intervention.
Run tokenizer/span preflight on every input, preserving exact input IDs for
anchors and all unchanged prompts. Freeze source, protocol, tests, data and
plans before submission. Use existing environments; no package installations.

Use unchanged inherited google/gemma-4-31B-it inference at revision
842da3794eaa0b77d5f08bae87a17459d91ff475, bf16 SDPA, thinking off, greedy batch
eight and 32 output tokens. A100 is required for historical and exact parent
replay. Each GPU job requests one A100, four CPUs, 96 GB RAM and one hour;
previous 7,664-generation baselines took 18–19 minutes. The largest diagnostic
is expected to take roughly 40–50 minutes including controls; shorter work is
aggregated into the context job. No artificial padding. Run frozen gpu_pick.py
with 75 GB VRAM, these CPU/RAM/time requirements, CUDA 13/x86_64 and no requeue;
save its best compatible A100 choice and override partition/QoS at submission.
No compatible placement means stop rather than change hardware. CPU preparation
uses one CPU/12 GB/20 minutes; dependent audit uses one CPU/8 GB/15 minutes,
normal QoS, no partition restriction and no GPU. Poll no faster than 60 seconds.

## Analysis and decision

Report every arm's accuracy, validity and truth/assigned/other/reader/compromise
choices by template, world, role, reminder and presentation. Pair each probe
with the same parent case in the unchanged full context; renamed labels compare
semantic row indices rather than literal labels. Report rescued and damaged
answers, reminder transitions, and all-three-state tracking. Report rebatched
parent answer/token differences separately from exact-batch anchor integrity.
Only combine results after all execution and sentinel/anchor audits pass.

A table-only failure would show the reporting incentives are not necessary for
that error. A full-context failure with a successful reduced context would
implicate the added context as a bundle, not uniquely identify a sentence or
latent mechanism. Row/column, encoding and numeric/lexical contrasts localize
behavioral dependence; they do not prove an internal parsing algorithm. Rescue
of token questions does not establish repair of private/payment controls.

No prompt-level significance tests, independent-sample interpretation of
rotations/repetitions, honesty inference or SOO claim. A concrete repair that
preserves the intended incentives could motivate a separately authorized fresh
revision with unchanged competence standards. If no robust repair emerges,
close Part 2 as a baseline limitation and choose a control-qualified Part 1
contrast for a separately frozen Part 3 protocol.
