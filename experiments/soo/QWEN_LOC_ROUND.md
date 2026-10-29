# Qwen localization round: where in Qwen3.8-27B does the L32 adapter's room-task effect live?

Written 2026-09-29, before launching. Exploratory. Brings Qwen3.8-27B to the point the gemma-4 and Mistral models reached through the layer round and the two bisection rounds ([FINDINGS2.md](FINDINGS2.md)), so that a band-steering round ([STEER_ROUND.md](STEER_ROUND.md)) can follow on a fourth model. The first study's Qwen3.8-27B LoRA at L32 is its cleanest cell: main and treasure hunt 100% honest against a 100% deceptive base in both orientations, at three seeds (FINDINGS.md, job 13710713). On the other three models the effect sits in a contiguous band of `v_proj` deltas below the training layer (12B 19–23, Mistral 5–13, 31B 20–31).

## What is different on this model

Qwen3.8-27B is a hybrid stack: 48 Gated DeltaNet (linear-attention) layers and 16 full-attention layers (3, 7, 11, …, 63). The adapter follows that split (`configs/qwen38-27b.yaml`): on full-attention layers it adapts `q_proj` and `v_proj`, and on DeltaNet layers it adapts the fused `in_proj_qkv`, whose output is [query | key | value] with 2048, 2048 and 6144 channels, followed by a per-channel convolution. The training layer L32 is a DeltaNet layer; the loss read its `linear_attn.out_proj` output at the last token. Below L32 there are 24 DeltaNet layers and 8 full-attention layers (3, 7, …, 31). Every LoRA B above L32 is exactly zero in the seed-0 adapter (read from the weights; recorded per module in `launch.json`), so there is no "above" condition.

"The `v_proj` band" therefore has two candidate analogs here: the value slice of the DeltaNet layers' fused projection, and the `v_proj` of the few full-attention layers. This round separates them, then bisects the layers below L32 with every module type kept.

## Intervention

As in the layer and bisection rounds: load the seed-0 adapter unmerged and swap every LoRA module outside the chosen set back for its frozen base layer (`selfconcept.soo.lora_subset`). New for the fused projection: a module spec with a slice suffix keeps the module but zeroes its `lora_B` rows outside the slice, so the delta s·B·A·x reaches only those output channels (`in_proj_qkv.v`: value channels only; `in_proj_qkv.qk`: query and key channels only). Because the following convolution is depthwise, the slices do not mix before the delta rule. Nothing is merged. `tests/test_soo_hybrid_subset.py` checks the slices on a tiny random Qwen3.5 model against the full adapter and the base.

Conditions, both orientations, all three scenarios, n = 250 (layers @ modules; "all" keeps every module type):

- `base`, `adapter-seed0`: same-day references.
- Coarse: `only-train` (32 @ all: the training layer's own `in_proj_qkv`), `below-train` (0–31 @ all).
- Module types below L32: `below-qkv` (0–31 @ `in_proj_qkv`, the 24 DeltaNet layers), `below-qkv-v` (0–31 @ `in_proj_qkv.v`), `below-qkv-qk` (0–31 @ `in_proj_qkv.qk`), `below-attn` (the 8 full-attention layers @ `q_proj`,`v_proj`), `below-attn-v` (the same @ `v_proj`).
- Bisection of 0–31 @ all: halves `lo-half` (0–15), `hi-half` (16–31); quarters `q1` (0–7), `q2` (8–15), `q3` (16–23), `q4` (24–31); `u20` (20–31, the relative position of the 31B band) and `top4` (28–31).

Collapse measurement as in the bisection rounds (`make_constants.py`: mean and spread of the L32 attention output at the last prompt token over the 156 training prompts) under each subset.

## Fixed setup

Qwen/Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded through `load_causal_lm` (multimodal checkpoint, text stack only), thinking disabled at every template call (`SOO_CHAT_KWARGS='{"enable_thinking": false}'`), bfloat16, one A100, greedy decoding, the first study's `room_only` suffix, 100 new tokens, scenarios `main` / `treasure_hunt` / `perspectives`, both orientations, rule-based classifier as primary readout with raw completions saved. DeltaNet layers run the reference PyTorch kernels (`fla` and `causal_conv1d` are not installed), as in the first study. Frozen snapshot with hashes, software checks, `submission.json`; complete runs are skipped on restart. The first study's evaluations of this adapter used n = 50 for treasure hunt and perspectives; the same-day references here replace them.

## Readout and decision, fixed in advance

The layer round's rule, unchanged: a condition **reproduces** when in both orientations its main deceptive rate is within 10 points of `adapter-seed0`, its refusal-plus-other rate is at most 10 points above the reference's, and perspectives correct is at least 90%; **damage** is refusal-plus-other above 20 points or perspectives below 80%; **null** is main deceptive within 10 points of `base` in both orientations; otherwise partial.

- `only-train` null and `below-train` reproducing: the gemma-4/Mistral pattern holds on a hybrid stack; go on to the module and bisection readouts. `only-train` reproducing would be the first model where the training layer's own module carries the effect.
- Module types: `below-qkv-v` reproducing and `below-qkv-qk` null is the value-path result of the other models, carried by the DeltaNet value channels; the band-steering round would then steer `in_proj_qkv`'s value slice. `below-attn-v` reproducing instead puts it on the eight full-attention `v_proj` modules. Both partial, or `below-qkv` and `below-attn` needed together, means the effect spans the two layer types; report it.
- Bisection: the smallest reproducing contiguous range among the halves, quarters, `u20` and `top4` is the band candidate. If only `below-train` reproduces, the effect needs all of 0–31 at this resolution; if a half reproduces but no quarter, a second bisection within that half follows, as it did on 31B.
- Collapse next to each verdict, as before; on the other models the collapse and the behavior dissociated.

No honesty claim follows from any outcome. The steering round on Qwen is written after this round's result, not before.
