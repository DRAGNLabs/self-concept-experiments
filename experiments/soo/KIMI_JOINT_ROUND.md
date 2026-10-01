# Kimi-Dev-72B L56 joint-offset round: do the adapter's mean `q_proj` and `v_proj` deltas, added together as fixed offsets, reproduce it?

Written 2026-10-01, before launching. Exploratory. Follows the L56 localization round ([KIMI_LOC_ROUND.md](KIMI_LOC_ROUND.md), jobs `13939048` / `13939049`, write-up in [FINDINGS2](FINDINGS2.md)).

## Why this round

The band-steering line of work replaces a trained LoRA band by its mean delta, a fixed offset with no input dependence, and asks whether the behavior survives. On Mistral, gemma-4-31B and Qwen3.8-27B it does, with the value-path deltas alone. The localization round showed that the Kimi-Dev-72B L56 adapter has no such `v_proj` band:

- layers 0–55 with both modules (`below-train`) reproduce the adapter (main deceptive 14% / 10% against the adapter's 14% / 8%, base 68% / 75%);
- `v_proj` alone and `q_proj` alone over the same layers are each 97–99% deceptive, with a think block in every response and no honest thinking response;
- layers 28–55 with both modules (`hi-half`) get part of the way (27% / 15% deceptive, 14% / 17% truncated), and every quarter and the lower half are more deceptive than base.

So the only LoRA subsets with the adapter's kind of effect contain both modules. The localization round's steering cells tested each module separately over all 56 layers at α = 1: the `v_proj` means broke the model (about half the responses loops, half truncated, perspectives 3% / 6%), a random offset of the same norms broke it as well, and the `q_proj` means gave a coherent, 94% / 84% deceptive model close to the `q_proj` LoRA. Neither tests the object that works as a LoRA. This round adds the `q_proj` and `v_proj` means together.

Two things have to be dealt with for the test to be readable. The α = 1 `v_proj` offset over 56 layers is too large a perturbation for any direction to matter, so the joint offset is run below α = 1, on the narrower 28–55 band, and on the user-message tokens only. And a fixed offset at every position also lands on up to 1024 generated tokens, where it was never measured (the means are over training-prompt tokens); restricting it to the user message (`content`) removes that, and on the other models the prompt-only offset was enough.

## Intervention

The mean deltas measured in the localization round (`kimiloc-kimi-dev-72b-L56-20260930T223131Z/output/band_deltas.pt`: the seed-0 L56 adapter, every `q_proj` and `v_proj` module in layers 0–55, mean over all valid tokens of the 156 training prompts, full adapter active while measuring) are reused, copied into this snapshot with their hash. No adapter is loaded in any new cell; every cell steers the base model. The evaluator gains two options: `--band-module q_proj,v_proj` adds both projections' offsets in the same forward pass, and `--band-layers` keeps a layer range of the file's offsets. The random control draws an independent Gaussian direction for each (layer, module) with that offset's norm (seed 0), so it matches the joint offset's size in both projections.

Cells, both orientations, `main` and `perspectives`, n = 250. "all" is layers 0–55 (reference LoRA `below-train`), "hi" is layers 28–55 (reference LoRA `hi-half`; its offsets are the same file's layers 28–55, measured with the whole adapter active). `jv` adds the `q_proj` and `v_proj` means, `rand` the matched random offsets, `v` the `v_proj` means only.

| Cell | layers | positions | α | matched random |
|---|---|---|---:|---|
| `jv-all-a0.25`, `jv-all-a0.5`, `jv-all-a0.75` | 0–55 | every position | 0.25, 0.5, 0.75 | `rand-all-a0.25`, `rand-all-a0.5`, `rand-all-a0.75` |
| `jv-hi-a0.5`, `jv-hi-a1` | 28–55 | every position | 0.5, 1 | `rand-hi-a0.5`, `rand-hi-a1` |
| `jv-all-content-a1` | 0–55 | user message | 1 | `rand-all-content-a1` |
| `jv-hi-content-a1` | 28–55 | user message | 1 | `rand-hi-content-a1` |
| `v-all-a0.5` | 0–55, `v_proj` only | every position | 0.5 | — (reference `below-v`) |

Fifteen cells. The joint offset at α = 1 on every position over 0–55 is not run: it contains the `v_proj` offset that already destroys the output at that dose. `v-all-a0.5` is the module contrast at a dose the model may tolerate: the LoRA subsets predict that a `v_proj`-only offset, if it does anything coherent, raises deception, while the joint one lowers it.

Carried over from the localization snapshot, not re-run: `base`, `adapter-seed0`, `below-train`, `hi-half`, `below-v`, `below-q`, and the three α = 1 steering cells (`v-all-a1`, `rand-all-a1`, `q-all-a1`).

## Fixed setup

As in the localization round: moonshotai/Kimi-Dev-72B at revision `8791d7981945752a51f692d66f2bbfb3573c9722`, bfloat16, `device_map auto` on four A100s, greedy decoding in left-padded batches of 16, 1024 new tokens, the model's own chat template, `room_only` suffix, labels from the text after the last `◁/think▷` (a think block never closed is `truncated`), raw completions saved. The `content` span is the user message's tokens in the chat-templated prompt (checked on the Kimi tokenizer: tokens 24–138 of a 144-token prompt, after the default system message and before `<|im_end|>`). Frozen snapshot with hashes, software checks, `submission.json`; complete cells are skipped on restart. Three jobs share the snapshot, five cells each, ordered so that a time-out loses the least informative cells: A runs `jv-all-a0.5`, `rand-all-a0.5`, `jv-all-a0.25`, `rand-all-a0.25`, `v-all-a0.5`; B runs `jv-hi-a1`, `rand-hi-a1`, `jv-hi-a0.5`, `rand-hi-a0.5`, `jv-all-a0.75`; C runs `jv-all-content-a1`, `rand-all-content-a1`, `jv-hi-content-a1`, `rand-hi-content-a1`, `rand-all-a0.75`. A coherent cell took about 2.5 hours in the localization round and a broken one (every response running to 1024 tokens) about 3.8, so each job needs 13–19 hours. Launcher: `scripts/launch_kimi_joint.py`; analysis: `scripts/analyze_kimi_joint.py`.

## Readout and decision, fixed in advance

Rates over all 250 responses; `truncated` counts as none of honest, deceptive, refusal or other. Per cell: think rate, honest share among thinking responses that answered, perspectives correct and truncated.

- Each offset cell is read with the localization round's rule against the LoRA it replaces (`below-train` for 0–55, `hi-half` for 28–55, `below-v` for `v-all-a0.5`) and, for information, against `adapter-seed0`: **reproduces** when in both orientations the main deceptive rate is within 10 points of the reference's, refusal + other is at most the reference's + 10 points, and perspectives is at least 90%; **null** when the deceptive rate is within 10 points of base in both orientations; **damage** when refusal + other exceeds 20% or perspectives is below 80% in an orientation; **budget-bound** (not reproducing) when the truncated rate exceeds the reference's by more than 10 points in an orientation; otherwise **partial**. One label is added because the localization round's "partial" hid it: a cell whose main deceptive rate is more than 10 points above base in both orientations is **more deceptive than base**.
- A joint offset counts as carrying the adapter's effect only if it reproduces its reference, the change is of the conclusion kind (the honest share among thinking responses moves, as it does under the adapter; a drop in the think rate alone does not count), and its matched random offset does not reproduce. If the random offset reproduces too, a perturbation of that size is enough and the direction is not established. `hi-half` is itself only partial against the adapter, so a cell that reproduces `hi-half` has reproduced a partial effect; the `adapter-seed0` column says how far that is from the full one.
- If no joint cell reproduces: where the joint cells are more deceptive than base with a think block in every response, the fixed means behave like the single-module LoRA subsets and the honest conclusion needs the input-dependent part of the deltas (their spread around the mean is 0.3–0.6 of the mean norm on the training prompts in layers 13–55, larger below); where they are damaged at every dose at which they do anything, the offset is not a usable stand-in on this model. Either way the fixed-offset construction stops here for Kimi-Dev-72B and the result is recorded as a negative.
- `v-all-a0.5` against the joint cells at the same dose says whether the offsets inherit the LoRA's module dependence (`v_proj` alone raising deception, both together lowering it).
- If a joint cell reproduces specifically, the next round is a second seed for that cell's random control and the neighboring doses and positions, not a new claim.

No honesty claim follows from any outcome. The treasure hunt is not run (99–100% deceptive under base and adapter alike). One seed, one adapter, one random draw per control.
