# Part 3: full-adapter prompt activation patches

Authorized 2026-10-29 following the completed crossed add-back diagnostic.
This is a separately frozen exploratory protocol. The original band failed
seed 1; query add-back rescues its reminder contrast but loses one seed-0
control. Only full qualifies across all seeds. Retain those failures and the
original thresholds. Part 2 remains closed at failed task validation.

## Target and cohort

Use the full adapters for seeds 0, 1, 2, every one of the 1,728 historical
prompts per seed, both exposed splits, original/reminder variants, both roles,
aligned/opposed goals, all worlds, table/answer rotations, and all four queries.
No case selection by output, fresh-task repair claim, new training, dose search,
or layer search. Use the pinned Gemma revision and parent A100/bf16/SDPA,
greedy batch-eight, thinking-disabled, 32-token generation configuration.
Preserve complete parent batches and their native tokens/left padding.

Patch outputs of all 110 actual full-adapter q_proj/v_proj modules jointly.
Retain the actual module names, including modules whose own updates are zero:
their outputs may carry upstream effects. This is a distributed intervention,
not a claim that each patched module is necessary or a unique mediator.

For each complete batch capture unpatched base and full-adapter projection
outputs on the identical native prompt, using the ordinary generation prefill.
Detach to host memory; do not retain or transplant donor KV caches. Finish and
discard donor generation before target generation. Target generation starts
fresh and constructs its own cache from the patched prefill. One model with
unmerged PEFT adapters switches through disable_adapter() for base. Exact
parent replays must validate this equivalence before scientific release.

## Fixed positions and conditions

Map character spans through the native tokenizer and freeze unpadded indices.
Record boundary-crossing tokens. Split the original first paragraph:

- **role:** first sentence, `You represent NAME.` only.
- **goal:** initial `your objective ...` clause plus the complete added goal
  reminder where present. The action-context prefix is excluded. Thus reminder
  trials have larger goal masks; report each variant separately.
- **facts:** the complete payoff-table block, including its heading and names.
- **later:** the final eight nonspecial user-content tokens (answer-format tail).
- **goal_control / role_control:** respectively the first N nonspecial tokens
  of the action-consequences paragraph, exactly matching each semantic mask's
  token count. These are task-relevant control text, not semantically neutral.
- **all_prompt:** every nonpadding input position, including BOS, turn tokens
  and assistant/thought header. Other masks exclude special/header tokens.

For each of these seven masks run base-to-full restoration and full-to-base
reverse patches. At the same goal and role masks, in each direction, also add
deterministic random directions matched to the donor-minus-current-target norm
separately for every example, module and token. Fixed random seed 1729; derive
streams from example/module/direction/mask, independent of batch ordering.
This yields 18 scientific conditions, plus unpatched base/full references.
No condition is dropped for invalid answers, control damage, or weak effects.

Patches act once at each projection during the initial prompt forward pass.
Generated-token calls are observed but returned unchanged; assert their length
is one, and assert each selected projection is patched exactly once. Exclude
left padding and check unchanged unselected entries. Record actual replacement
L2 norms per layer/module/example/token, with summaries and counts. Random
matching uses float32 directions before bf16 rounding. If needed, calibrate
the scalar magnitude by 18 bisection steps on [0,4] to compensate for rounding,
without changing the sampled direction or consulting answers. Log realized and
desired norms and check the discrepancy (1% relative or 1e-5 absolute).
Position, span length, lexical content and downstream information remain
confounded. Literal-span nulls do not exclude information at other positions.

## Mechanical qualification and release

CPU Slurm tests cover left-padding indices, identity/replacement semantics,
unselected/decode preservation, random norms/determinism, hook cleanup on error,
and mask construction on all 1,728 native inputs. Freeze source, data, adapters,
protocol, plans and tests before any new GPU output.

Three aggregate GPU mechanics jobs run all complete historical batches and
parent sentinels. In every batch, reproduce unpatched base and full raw tokens
exactly; capture donor outputs and test all-prompt base-to-base and full-to-full
identity against those references. Also run synthetic numerical hook tests on
the allocated GPU. No cross-model scientific patch is run at this stage.
Repeat before/after sentinels and exact historical regressions. A dependent CPU
audit independently checks all three seeds, native inputs, module names, source,
versions, hashes, coverage and identity equality. Any failure stops release;
no automatic retry. Only after every mechanical audit passes, consult gpu_pick
again and release three frozen scientific seed jobs and their final audit.

Scientific jobs repeat baseline capture and exact reference checks before each
batch's patches, then all 18 conditions. Retain before/after sentinels in every
condition and audit exact repeats. Sources and analysis use the same frozen
snapshot, unaffected by live-tree merges. No cached source activations persist
to shared disk. Request A100 >=75 GB, 4 CPUs, 96 GB RAM, CUDA 13, x86_64,
no preemption: one hour for mechanics, three hours for scientific jobs. Save
gpu_pick evidence and measure runtime/memory; no further experiments auto-release.

## Analysis and limits

Report absolute assigned-client and recipient-advice rates, validity/truncation,
private/factual correctness, paired transitions against the target reference,
and unconditioned joint assigned-advice recovery with all three controls correct.
Keep aligned cases and report by seed, split, variant, world and role. Compare
goal/role effects directly with token-count-matched controls, factual/later
positions and random perturbations. Summarize actual per-token norms as well as
counts; nominal masks alone do not establish specificity. Report all-prompt
restoration descriptively: full-adapter decoding remains active, so base-equivalent
generation is not guaranteed. Reverse patches test the converse intervention.
No prompt-level p-values, independent-world claims for rotations, unique-mediator
claims, general honesty claims, or untouched-holdout language. Behavioral damage
under a patch is a result; mechanical/integrity failure prevents interpretation.
