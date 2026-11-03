# Part 3 band discrepancy: crossed add-back diagnostic

Authorized continuation of the existing-data audit and Part 3; the user asked
to investigate the failed band qualification on 2026-10-29. This specification
is frozen before new add-back outputs. Exploratory, previously exposed cases.

The completed audit found that all historical controls survive the band. Seed
1's late-reminder recipient advice rises from 9/54 to 25/54 in development and
7/54 to 27/54 reserved. In the band, role 0 gives 24/27 in both splits versus
1/27 and 3/27 for role 1. Role 0 is also the first payoff column: these results
do not distinguish role identity from column conventions. Differences occur
in every historical world and presentation rotation. All 18 integrity audits
and a 66-module subset identity test pass. All weights are finite and retained
updates nonzero; seed 1's band weight norm lies between seeds 0 and 2. These
checks do not establish a neural mechanism or prove complete absence of bugs.

## Fixed conditions

Keep the existing band B = v_proj at actual layers 20–31. Cross two add-backs:

| Condition | Query-projection updates | Value updates outside B |
|---|---|---|
| band | Absent | Absent |
| add_q | All restored | Absent |
| add_v | Absent | All restored |
| full | All restored | All restored |

This is four cells of a fixed 2×2 design, not a layer/dose sweep. The actual
adapter contains 60 q_proj and 50 v_proj modules; B has ten. Training stopped
at layer 32: updates above layer 32 are exactly zero in all seeds. The
nonzero outside-B value updates are therefore below layer 20 and at layer 32.
Retain zero-update modules where the condition specifies all; report their
presence explicitly. No merging or changes to surviving LoRA tensors.

Run **all 1,728 historical prompts per seed and condition**, retaining both
splits, original/reminder, both roles and alignments, every table/answer
rotation, and private/message/both factual queries. Preserve complete parent
batches and native inputs, including when concatenating the two splits.
No cases or seeds are selected by outcomes. The two new conditions contribute
10,368 answers; exhaustive same-condition full/band replays contribute another
10,368. Fresh-template data are not claimed to be repaired by this diagnostic.

Use the frozen Gemma revision, code, bf16 SDPA, thinking disabled, greedy batch
eight, 32-token cap, A100 and original environments. Each seed job first runs
and audits full and band replays against their exact historical parent raw
answers/tokens, then both add-backs. Whole-batch sentinels repeat before and
after every condition. Run native-input, hash/version, actual-module, coverage,
completion and historical-regression checks; regression outputs for new
conditions are descriptive. No activation patches are launched by this job.

CPU preflight tests module selection and exact retained/removed forward
outputs, freezes the new controller, and runs gpu_pick for one A100 ≥75 GB,
four CPUs, 96 GB host RAM, CUDA 13, x86_64, one hour, no preemption. Three seed
jobs aggregate all four conditions; final CPU analysis depends on all three.

## Interpretation

Report paired answer transitions and absolute recipient-advice rates by seed,
split, variant, role, world, alignment and query, plus all private/factual
controls and invalid outputs. Report the descriptive factorial interaction
`full - add_q - add_v + band` in rates, separately for each cell; no prompt-level
significance tests, and rotations are not independent worlds.

Apply the **unchanged** historical feasibility thresholds from PART3_CAUSAL.md
to each condition across all three seeds, retaining failures. A query-only
add-back restoring the contrast supports a contribution from the query updates
in the presence of B; an outside-value add-back restoring it supports that
value contribution. If only full restores it, the combination is required
under these interventions. Neither establishes a unique mediator or semantic
representation, and equal behavioral rates do not establish equal computation.
Inspect both controls and case-level agreement with full, not only gate passes.
Report any tradeoffs across seeds. Do not automatically choose a winning
condition as the patch target or launch a further search: interpret the fixed
factorial comparison first. Part 3 remains authorized; its target must follow
the evidence and be specified before activation-patch outcomes are observed.
