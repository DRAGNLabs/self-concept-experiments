# Synthetic reward-hacking axis: plan

Written 2026-11-03, before any run. This plan goes with the persona and reply-pair comparisons on branch
`rh-axis-geometry`. Per Koby, there is no FINDINGS entry for now. Results are reported in chat, and the memory file
`rh-axis-vs-gi-v2-geometry` records them.

## Question

Goodfire built a reward-hacking direction from scripted, matched hack/honest conversations
(arXiv 2609.19101, Sec. 3.1 and App. B). Two questions:

1. Applying their method and data to gpt-oss-120b and Muse-Glimmer-30B, what direction do we get?
2. Rebuilding it for the hacks our minimal harness actually allows, what direction do we get?

For both, we ask where the direction sits relative to:

- the deceiver persona axis (gi_v2) and the 17-role gi;
- the letter-vs-spirit axis, its six pairs, and the jobsworth control;
- Chris's reply-pair axes.

We also ask whether it separates real hack replies from their honest twins better than those axes do. Steering
comes later and is not in this plan.

## Goodfire's data (what we know)

- **Release:** HF `Goodfire/reward-hack-data`, folder `synthetic_data/`. Copy downloaded 2026-11-03 to
  `~/nobackup/autodelete/self-concept-experiments/goodfire_rh_synth/hf/`. There is no official code.
- **Size:** 3,123 conversations (2,504 train, 619 validation), in 4 of the paper's 5 families. The reporting family
  (2,162 conversations) is not released.

  | Family | Variants per task | Count |
  |---|---|---|
  | test-suite tampering | honest, rm_test, edit_test, conftest_ignore, env_var | 200 × 5 |
  | validator tampering | honest, print_line | 200 × 2 |
  | score-file tampering | honest, write_score | 200 × 2 |
  | information acquisition | honest, peek and exploit × {hidden tests, fixture, reference} | 189 × 7 |

- **Pairing:** variants of one task differ only in the action message. The tool result and the final report are
  identical.
- **Format:** standard HF messages: assistant `tool_calls`, `role: tool` results, thinking off, no tool schemas.
  gpt-oss's and Muse's chat templates both render these natively; this was checked by rendering one conversation in
  each.
- **Readouts:** each readout is stored as `{message_index, suffix}`. On Qwen it is the 6 closing tokens
  `</function>\n</tool_call><|im_end|>` of the act, read or patch tool call, averaged.
- **Contrasts:** `contrasts.json` defines 18 hack-vs-honest comparisons.
  - individual `i_*`: one hack variant against honest;
  - pooled `pl_*`: several hack variants together.

  The exploit comparisons subtract peek, not honest.
- **Unknowns:** the paper's "Projected" rows are not released and not defined.
- **Monitoring:** on real rollouts the paper scores every token with ⟨v, x_t⟩. Thresholds are set to match the LLM
  monitor's FPR. Ground truth is per passage, and the peak score within a passage is used, so we aggregate with a max.

## Engineering

Three small commits, each reviewable on its own. Then the extraction script.

1. **`batch_conversations` refactor.** Move the hook-and-forward part of `ActivationExtractor.batch_conversations`
   into a method that takes token ids. `batch_conversations` calls it and behaves exactly as before.
2. **Narrow protocols.** These go in a new module; the existing `ModelSpecifics` protocol is unchanged.
   - `MessageSpanSpecifics.message_spans(tokenizer, full_ids) -> list[MessageSpan]` gives every rendered message:
     role, kind (reasoning / content / tool call / tool result), content token range and end-of-message token.
   - `ToolCallCloseSpecifics.tool_call_close(tokenizer, span)` gives the closing-markup tokens of a tool-call message.
   - `ReadoutSpecifics(MessageSpanSpecifics, ToolCallCloseSpecifics, Protocol)` is the intersection.

   gpt-oss, Muse and Gemma 4 implement these by wrapping their existing message parsers. They satisfy the protocols
   structurally, without subclassing them.
3. **Readout module.** Each readout function asks for exactly the protocol it uses:
   - `last_token_readout(specifics: MessageSpanSpecifics, …)`
   - `all_response_readout(specifics: MessageSpanSpecifics, …)`
   - `close_readout(specifics: ReadoutSpecifics, …)`

   The model name is resolved to a specifics object once, at the top level of the script (`specifics_for`). It is
   never resolved inside library functions.

Readout variants, all extracted at every layer:

| Variant | Tokens |
|---|---|
| close | Goodfire's choice: closing markup of the action message, averaged |
| last | the action message's end-of-message token |
| all response | mean over every assistant token in the conversation, matching how the persona vectors are built |

## Step 0: visualization script (CPU, existing results)

One script reads the geometry and logit-lens JSONs. Later directions show up in the same plots just by adding their
entries. The figures, per model:

1. A cosine heatmap of all directions at the steering layer. It uses a diverging scale centered at 0, with values
   printed in the cells and cells inside the sign-flip noise level greyed.
2. A layer × direction heatmap of cosine with gi_v2, and a second one with Chris's axis.
3. Separability:
   - a per-pair dot plot of hack-minus-honest projections on Chris's pairs, with a line at 0 and k/n printed;
   - AUROC with bootstrap 95% intervals, both models on one axis.
4. A scatter of each letter/spirit pair's gi_v2 cosine against its pair accuracy on Chris's pairs, both models.
5. Logit lens: for each direction and layer, bar charts of z for the top and bottom 8 tokens, with a line at the
   random-direction 95th-percentile |z|.

Charts follow the `dataviz` skill, and the palette is checked with its validator.

## Step 1: Goodfire's data on gpt-oss and Muse (GPU)

- Render every conversation with the model's chat template, using the chat settings from Chris's pair runs.
- Extract all three readouts at every layer, one job per model, about 15 minutes. Placement comes from `gpu_pick`,
  logged in `jobs.txt`.
- Build every comparison in `contrasts.json` on the train split, and check it on the validation split.
- Run the test battery below, and add a family × family cosine heatmap (their Fig. 17).

## Step 2: our own synthetic pairs

### Format: the minimal harness, not tool calls

Our harness has no tools. The model replies with a fenced code block, and the harness returns the test output as the
next user turn. Transcripts are built with `selfconcept.codebench.harness`'s own message builders, so the format is
exactly what the model sees in real episodes:
- `check_task_message`, check kind, with the prompt C / no-carve instruction and the neutral stated budget;
- `neutral_retry_message`.

There is no reasoning (as in Goodfire's data), so the action message is the only difference within a pair. The
"close" readout here is the code block's closing fence plus the end-of-message token. That needs only
`MessageSpanSpecifics`.

Every variant has the same three turns:

1. task + visible `check()` asserts → first reply with the buggy solution;
2. neutral harness output showing 1–2 failing asserts;
3. **the action**: a second reply whose code block differs by variant.

### Families (the turn-3 code block)

| Variant | Turn-3 code |
|---|---|
| honest | the correct solution, which contains ≥ 1 general conditional return and the visible input/output pairs as `assert solve(x) == y` self-checks |
| hardcode | the buggy solution kept as a fallback, behind a table of the visible tests' expected outputs as the main path, with the same literals as honest |
| special_case | the buggy solution plus `if <input == failing test input>: return <expected>`, in the same style and place as honest's general conditional return |
| rm_test | the buggy solution plus `check()` redefined without the failing asserts |
| edit_test | the buggy solution plus `check()` redefined with the failing asserts' expected values changed to the buggy outputs |
| replace_check | the buggy solution plus `def check(candidate): pass` |

Both hardcode and honest contain an algorithm and the same literals. In hardcode the literals are the main path,
and the failing output in turn 2 shows the model knows the algorithm is wrong. honest and special_case both contain
a conditional return; one is general and the other is keyed to an exact input.

The three tampering variants are not form-matched. Goodfire's are not either.

### Generation

- **Problems:** invented, easy-to-medium, single-function problems. We don't use LCB problems because we have no
  reference solutions for them, and the direction should capture the hack, not the difficulty.
- **Who writes them:** Sonnet subagents. One subagent writes each task as one unit:
  - problem statement;
  - correct solution, with ≥ 1 general conditional return;
  - buggy solution, with one realistic bug;
  - about 5 visible tests;
  - a brute-force reference;
  - an input generator for hidden tests.

  A deterministic script assembles every variant from these. Subagents never write transcripts.
- **Varied conditions:** the general conditional return is spread across categories: base case, empty or boundary
  input, parity, off-by-one in a range, modulo/overflow. Each category is at most about 20%. Where honest uses
  recursion, the hack variants keep the same recursive function, so recursion cancels in the difference.
- **Scale:**
  - pilot: 1 subagent writes 10 tasks;
  - Koby reviews a sample;
  - full run: 10 subagents × about 28 tasks each, about 280 tasks, aiming for about 200 valid (about 1,200
    transcripts).
  - Each subagent gets distinct topic areas and category quotas. A duplicate check runs afterwards.

### Validation

Every task's code runs in a CPU SLURM job; none runs on the login node. A task is kept only if all of these hold:

| Check | Must hold |
|---|---|
| reference | correct solution == brute force on random small inputs; the correct outputs become the hidden tests |
| buggy | fails exactly the asserts shown in turn 2; the turn-2 text is the real harness output |
| honest | passes visible and hidden tests |
| each hack variant | passes the visible `check()` as the harness runs it, and fails the hidden tests |
| matching | variants differ only in the turn-3 code block (diff check), with similar lengths |
| conditional return | honest's code has a general conditional return of the assigned category (AST check) |

After that:
- **Blind judge:** our RH judge labels each transcript without knowing its variant; disagreements get inspected.
- **Hand review:** about 20 random pairs.

## Test battery (Steps 1 and 2)

The same tests as the earlier comparisons, at every layer, highlighting gpt-oss L17/18 and Muse L26:

- cosine with gi_v2, the 17-role gi, letter-vs-spirit and its six pairs, jobsworth and Chris's axes, against the
  sign-flip null;
- split-half reliability over tasks;
- consistency across families (family × family cosines);
- pair accuracy and AUROC on Chris's real hack/honest reply pairs (gpt-oss same_prompt and stage C; Muse stage C);
- logit lens (bar-chart form).

Step 2 also checks for confounds:

- **leave-one-category-out:** a direction rebuilt without one condition category should stay close to the full one;
- **hardcode vs. special_case vs. tampering:** how much do the families share?
- **logit lens:** no recursion- or category-specific tokens.

## Later

- **Per-token traces:** score every token, take the max per passage, and plot the score through full real episodes
  with the hack span shaded. This needs full episode transcripts, e.g. the stage-C episodes behind Chris's pairs, if
  they are available.
- **Real-episode transfer set:** pair real failed attempts with templated hack versions. Built only after the invented
  set works.
- **Steering with the new directions.**

## Outputs

Scratch root: `~/nobackup/autodelete/self-concept-experiments/schemer-axis/synthetic_rh/`:
- `goodfire/<model>/` for Step 1;
- `ours/{tasks,validated,transcripts,<model>}/` for Step 2;
- `figures/`.

Each model root has its own `jobs.txt` and `gpu_pick` output.
