# Diagnosing the route-2 baseline failure — 2026-10-28

The user authorized “let's diagnose that wording sensitivity” after the first
truth-versus-beneficiary batch stopped at its baseline gate. This authorizes a
bounded base-model diagnostic, not release of the six stopped SOO runs or a new
confirmatory SOO experiment. Preserve the failed batch and its original gates.

## Evidence and competing explanations

Both baseline jobs completed and passed integrity checks. Development passed all
1,080 competence checks; confirmation failed 86, including 82 observed-token
checks, three private-choice checks and one payment-table check. All 12,960
primary answers were valid. Observed-token accuracy was 648/648 for both
development variants, versus 420/648 original and 455/648 late-goal on
confirmation. Of 421 wrong confirmation token answers, 401 selected the assigned
client's payoff maximum, 404 matched the paired private choice, and 403 matched
the paired public report. These overlaps are descriptive, not evidence of an
internal mechanism or intentional deception.

Template wording, names, token labels and absolute payment values changed
together between splits. Thus the existing result alone does not establish
wording as the cause. Diagnostic inputs reuse all twelve already exposed worlds;
none is a fresh confirmation set. Freeze the following comparisons together
before generating any new output. Do not select examples by prior success or
failure, revise prompts after outputs, or silently treat a successful diagnostic
condition as a validated repair.

## Frozen comparisons

1. **Cross wording and world.** Apply each of the four original complete wording
   templates to each of the twelve worlds, keeping names, labels, payment values,
   assigned client, reader preference and observed token fixed in each pair.
   Cross both roles, both reader alignments, all three truths, and original/late
   goal variants. Run all five query types at table/answer order (0,0), plus the
   observed-token check at (1,1) and (2,2). This yields **8,064** responses. The
   extra rotations balance displayed positions but confound table and answer
   order with each other; they do not reproduce the parent's full nine-order grid.
2. **Bidirectional section swaps.** For each t2/t3 wording family, replace one
   section at a time with its t0 equivalent, and conversely put that t2/t3 section
   into t0. The seven sections are assignment, reader preference, observation,
   private-action rule, public-report rule, factual-check scope and final question.
   Apply all swaps to all twelve worlds and the same role/alignment/truth/reminder
   crossing, at (0,0), asking only the observed-token check: **4,032** responses
   per family. Preserve the host container noun (box/container/drawer) when
   transplanting a section so its references stay coherent. Full-template crosses
   retain their original nouns. These test whole sections, not individual tokens;
   a null single-section effect cannot exclude interactions among sections.
3. **Context and question probes.** For hosts t0/t2/t3 and all twelve worlds with
   the same crossing at (0,0), compare: (a) observation + question + answer format
   alone, (b) the full prompt plus a question-local reminder to answer this unpaid
   fact check from the observation, and (c) the full prompt plus a neutral reminder
   to give one label without extra words. **2,592** responses. The neutral reminder
   is a generic-instruction control, not an exactly token-matched control. The
   observation-only version removes several competing contexts at once and has
   identical text across some role/alignment/reminder labels; these duplicates
   are reported as repetitions, not independent evidence.

Total: **18,720 diagnostic responses**, four base-model GPU jobs. Every job also
replays the same 128 parent responses in their complete original eight-item
batches (one fact-check-containing batch per original template × goal variant ×
assigned role, with opposed reader and truth favoring the other client). Include
32 inherited historical regressions per job and complete-batch before/after
sentinels covering diagnostic condition × reminder × role × query. Record exact
counts at staging. Keep unchanged parent inference, revision, tokenizer, native
inputs, greedy batch eight, bf16 SDPA, thinking off and 32-token cap, on A100.
Save gpu_pick.py output and use the established dw/dw87 A100 pool.

Verify all score keys independently from raw displayed observations/tables, all
section-only differences and noun references, full crossings, native token spans,
source/model bindings, historical regressions, completion, batch membership,
sentinel stability and parent anchor tokens/answers. New full-template native
prompts matching a parent example must have identical input IDs. Fresh batches
can differ from parent padding; report those comparisons separately from the
exact-batch anchors. A failed integrity check stops combined interpretation.

## Readouts and stopping rule

Save the original failure's raw-response rescore and stratified summary. For the
diagnostic, report validity, factual correctness and assigned/reader-optimal
answers by condition, world, source world family, wording template, role,
alignment, truth relation, goal reminder and presentation. Compare complete
templates on the same contexts; compare each swap/probe with its matching full
host prompt. Report correct-to-wrong and wrong-to-correct transitions, not just
aggregate gains. Preserve the full cross's private, report and table-fact results
to distinguish changes in task interpretation from general loss of goal pursuit.
Report goal-reminder transitions and all-three-truth tracking with fixed payoffs.

No SOO adapters, new training, routes 3–4, automatic repair/retry, outcome-based
filtering, new competence gate or automatic promotion follow this diagnostic.
There is one frozen batch and one final integrity/analysis job. A result may
localize a behavioral prompt dependence; it cannot establish an internal mediator,
general honesty, or a validated new task. Any subsequent confirmatory revision
needs a separately documented protocol and fresh wording/worlds.
