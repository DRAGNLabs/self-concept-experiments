# Steering-vector and LoRA results matrix

Tracking file: per-model numbers for both SOO interventions — steering
(add-mode, o_proj, best validated cell unless noted) and LoRA
fine-tuning (best validated recipe). Narrative and analysis live in
FINDINGS.md; this file is numbers only. Goal: fill an age × size matrix
of models to see how each intervention trends with each variable.

## Model registry

| model | release | params | layers | hidden | working layer | rel |v|/|act| |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 2023-12 | 7B | 32 | 4096 | L16 (LoRA band) | 0.89 |
| Gemma-2-27b-it | 2024-06 | 27B | 46 | 4608 | L14 | 0.24 |
| OLMo-2-1124-7B-Instruct | 2024-11 | 7B | 32 | 4096 | L19 | 0.11 |
| Muse-Glimmer-30B | 2026-08 | 30B | 52 | 6656 | L26 | 0.15 |
| gemma-4-31B-it | 2026-07 | 31B | 60 | 5376 | L30 | 0.10 |
| gemma-4-12B-it | 2026-07 | 12B | 48 | 3840 | L19 | 0.13 |
| Llama-2-70b-chat | 2023-07 | 70B | 80 | 8192 | L16 (pilot) | 0.61 |
| Qwen2.5-72B-Instruct | 2024-09 | 72B | 80 | 8192 | L40 (pilot) | — |
| Kimi-Dev-72B | 2025-06 | 72B | 80 | 8192 | — | — |

## Headline results

Honest % on main scenario (base → best steered cell), n=50 unless noted.

| model | cell | main orig | main mir | TH orig | TH mir | persp | rand ctrl | n=250 | damage edge | caps Δ (ARC/HS/MMLU) |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B | L16, any α | 8 → 8 | — | 2 → 2 | — | 100 | matches real @α8 | — | α=8 (generic) | — |
| Gemma-2-27B | L14 α=32 | 0 → 48 | 0 → 36 | 0 → 0 | 0 → 0 | 100 | null @α16 | 42% @α16 | none ≤α96 | +0.4/−0.7/−2.5 @α32 |
| OLMo-2-7B | L19 α=16 | 86 → 26 | 6 → 0 | 0 → 0 | 0 → 0 | 100 | partial (86→66) | — | none (anti-honest) | — |
| Muse-30B | L26 α=8 | 2 → 90 | 58 → 100 | 0 → 12 | 0 → 42 | 100 | null @α16 | 94% @α8 | none ≤α32 | −2.1/−1.1/−1.7 @α8 |
| gemma-4-31B | L30 α=16 | 0 → 98 | 0 → 44 | 0 → 92 | 0 → 8 | 100 | 27% @α16 n250; mir 0% @α16 | 93% @α16 | none ≤α32 | +0.3/−1.2/−2.9 @α16 |
| gemma-4-31B | L30 α=20 | 0 → 100 | 0 → 94 | — | — | 100 | 80% orig / 16% mir | — | none | — |
| gemma-4-12B | L19 α=12 | 0 → 100 | 0 → 100 @α32 | 0 → 100 @α32 | 0 → 100 @α32 | 88–90 | 99.2% n250; −v 100% | 100% @α32 | none ≤α32 | — |
| Llama-2-70b | L16 α=8 | 64 → 92 | 6 → 72 | 12 → 100 | 0 → 94 | — | orig 80–86 + −v 98 (agnostic); mir 6 (null) | 89.6 orig / 74.0 mir | α=32 (L24/L32 degenerate) | — |
| Qwen2.5-72B | L40 α=16 | 0 → 100 | 0 → 100 | 0 → 100 @α32 | 0 → 100 @α32 | — | mir: rand 28/0, −v 22 (clean); orig: rand 8–72 ×4, −v 88 | 100.0 orig @α16; 98.8/99.6 @α32 | α48 → 74; `other` 0 everywhere | — |
| Kimi-Dev-72B | L24 α=32, 512 tok, post-think reclass* | 36 → 98 | 20 → 100 | 0 (base) | 0 (base) | — | pending (13573967) | pending | none; think channel deleted (33/50 → 0/50) | — |

Verdicts: Mistral **inert** · Gemma-2 **partial** (saturates ~60–66 orig /
~45 mir) · OLMo **harmed** · Muse **strong** · Gemma-4-31B **strong at
α=16** (direction-specific: real 98% vs rand 18%/4%; α≥24 is a second,
direction-agnostic flip regime — any matched-norm perturbation except −v
flips it; mirrored/TH/n250 numbers above were taken at α=32 in that
confounded regime, re-anchor at α16 = job 13562348) ·
Llama-2-70b **strong (mirrored-validated)**: orig orientation is
direction-agnostic (randoms 80–86, −v 98, from the matrix's highest
baseline of 64), but mirrored is clean — baseline 6, rand 6, real
72/74 n250, TH 0→94. Same dual-regime as the 31B, decided the same way. ·
Qwen2.5-72B **strong (mirrored-validated, direction-specific)**: at
L40 α16 mirrored, real 100 vs rand 28/0 and −v 22; orig orientation is
sign-agnostic (−v 88, rand 8–72 across 4 seeds) — same
orientation-contamination pattern as Llama-2-70b, decided the same
way; n250 real α16 orig = 100.0; above α24 fully agnostic, L16 fully
agnostic · Kimi-Dev-72B **steers 36→98 / 20→100, controls pending**
(13573967): \* reasoning model — at 100 tokens the eval is invalid
(33/50 truncated mid-CoT; classifier scores think text; truncation
biased the answered subset *honest* because deceptive answers
deliberate longer). All numbers from 512-token reruns reclassified
post-think. Steering deletes the think channel (33/50 → 0/50).

## Dose curves (main orig honest %, n=50)

| α | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2 L14 | 0 | 0 | 0 | 18 | 40 | 44 | 48 | 58 | 60 | 66 |
| Muse L26 | 14 | 34 | 80 | 90 | 92 | — | 92 | — | — | — |
| gemma-4 L30 | — | — | — | 0 | 98 | 100 | 100 | — | — | — |
| g4-12B L19 | — | — | — | 0 | 100 | 100 | 100 | — | — | — |
| Llama70b L16 | — | 60 | 58 | 92 | 70 | — | 20 | — | — | — |
| Qwen72B L40 | — | — | 0 | 38 | 100 | 100 | 100 | 74 | — | — |

Qwen72B L40 α12 = 92; rand-s0 dose at L40: α4 0, α8 0, α12 12, α16 58,
α24 100, α32 100 — randoms lag real by ~one dose step, then catch up.

12B fine-grained: α10 = 62, α12 = 100. 12B specificity: −v(α12) = 100,
−v(α32) = 100, rand s0(α12) = 68, rand s1(α12) = 0 — direction-agnostic.
31B mirrored dose: α16 = 44 (rand 0), α20 = 94, α24 = 98 (rand 58).
| OLMo L19 | 86 | 84 | 82 | 50 | 26 | — | — | — | — | — |
| Mistral L16 | 6 | 8 | 6 | 0* | — | — | — | — | — | — |

\* degenerate text (damage), not honesty. OLMo α=1–4 values approximate
from pilot grid. Gemma-4 α=12 → 6. Gemma-4 layer sweep at α=32:
L12/L18/L24/L27/L33/L36 = 0, L30 = 100. Random matched-norm at L30 α=32
also = 100 (specificity unresolved, job 13562331).

## LoRA (SOO fine-tuning) headline results

Honest % on main (base → best validated recipe). Mirrored baselines
differ per model (positional confounds); "conf." = scenario is
position-confounded on that model, rate not meaningful.

| model | recipe | main orig | main mir | TH orig | TH mir | persp | caps Δ (ARC/HS/MMLU) | damage character |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B | L16 lasttok lr1e-4, 5 seeds n250 | 10 → 90 ± 6 | 18 → 100 | conf. | conf. | 100 | −2.5/−0.7/−0.8 (worst seed) | evasion/scramble off-band; L19 evasive basin −10 ARC |
| Gemma-2-27B | L14 paper recipe, 2 seeds | 0 → 92/78 | 0 → 86/66 | 0 → 22–42 | 0 → 78/72 | 100 | — | deflection/refusal at L20/L28/L34; confabulation at L23 |
| OLMo-2-7B | split: lasttok L16 / full L19 | 86 → 90 (L16) | 6 → 70 (L16) | 0 → 4–26 (L19: 0→95) | 0 → 96 (L19) | 96–100 | — | moralizing refusal at L22; main effect positional at most layers |
| Muse-30B | none (10 layers × 25–98% depth; r64, allmod, r64allmod) | 4 → 4–36 (positional/confab) | — | 0 → 0 | 0 → 0 | degrades at L26 | — | no band: no-op → echo/degeneration, nothing between |
| gemma-4-31B | L32 Gemma-2 recipe, 3 seeds | 0 → 100/100/100 | 0 → 100/100/100 | 0 → 100/100/92 | 0 → 96/100/82 | 100 | — | n250 seed0: 100.0 orig / 100.0 mir; band razor-sharp: L30 0–50 seed-fragile, L34 refusal wall |
| gemma-4-12B | L24 Gemma-2 recipe, 3 seeds | 0 → 92/98/74 | 0 → 86/90/58 | 0 → 100 | 0 → 98–100 (s2: 74) | 76–96 | — | model intact; validated both orientations; n250 seed0: 94.4 orig / 92.0 mir; L19 near-inert (main 8) |

31B L26 is a genuine TH-only band (TH orig 94 / mir 96, main 0 both).
31B sweep, main orig: L18 0, L26 0, L28 2, L30 50 (s1/s2: 0), L32 100,
L34 2 (98% refusal). 12B sweep, main: L14 = 48, L19 = 8, L24 = 92. 12B
TH 0→98 *mirrored* — the OLMo-style positional confound is ruled out.
LoRA bands sit at ~50% depth on gemma-4 (12B L24/48, 31B L32/60) vs 30%
on Gemma-2 — sweep past 50% before declaring a cell dead.

LoRA verdicts: Mistral **strong** (validated, n250, mirrored) ·
Gemma-2 **strong** (validated, both orientations, genuine TH) ·
OLMo **partial** (no single recipe gets both scenarios; main confounded) ·
Muse **none** (no honest regime at any depth or strength) ·
gemma-4-12B **strong** (validated: 3 seeds, both orientations, genuine
TH, n250 94/92) · gemma-4-31B **strong** (validated: 3 seeds, both
orientations, n250 100.0/100.0 — cleanest LoRA cell in the matrix).

## Age × size matrix (steering / LoRA)

Verdict per cell, both interventions: steering = validated
direction-specific honesty gain; LoRA = validated honesty band.

| | ~7B | ~12B | 27–31B | ~70B |
|---|---|---|---|---|
| **2023** | Mistral-7B: **inert / strong** | — | *open / open* | Llama-2-70b: **strong / open** |
| **2024** | OLMo-2-7B: **harmed / partial** | — | Gemma-2-27B: **partial / strong** | Qwen2.5-72B: **strong / open** |
| **2025** | — | — | — | Kimi-Dev-72B: *steers 36→98, controls queued (13573967)* |
| **2026** | *open* | gemma-4-12B: **agnostic-flip / strong** | Muse-30B: **strong / none** · gemma-4-31B: **strong / strong** | *open* |

Trend, after the 70B validation and 31B round 3: **no axis predicts
LoRA failure** — strong on 2023/2024/2026, small and large; the sole
"none" is Muse-30B, a single-model (lineage) outlier. **Steering's axis
is capability/scale, not age**: a 2023 70B steers (mirrored-validated,
rand-null), so the 2023–2024 failures reflect those models all being
≤27B. The live contrast: where both
work, LoRA reaches a higher, more symmetric ceiling (31B: 100/100 vs
steering's orientation asymmetry), but finding its band cost three
rounds of layer search (bands sit at ~50% depth on gemma-4 vs 30% on
Gemma-2) vs one extraction pass for steering — steering's advantage is
search cost, not ceiling. All three ~70B models now flip 0→~100, so at
scale the question is no longer *whether* honesty moves but whether the
SOO direction is doing the moving: Llama-2-70b is direction-specific
(mirrored: real 72–74 vs rand 6), Qwen2.5-72B likewise (mirrored at
L40 α16: real 100 vs rand 28/0, −v 22 — the orig orientation's
sign-agnosticism was orientation contamination, as on Llama),
Kimi-Dev-72B awaits controls. "Steerability tracks scale" holds, and
both mirrored-validated 70Bs are direction-specific; whether
gemma-4-12B's "agnostic-flip" survives its own mirrored controls
(13575634) decides if that class exists at all. Kimi also sets a standing protocol rule: reasoning models need
≥512 tokens + post-think classification (at 100 tokens truncation
biased its answered baseline *honest* — the confound can point either
way). In flight: 12B mirrored controls (13575634), Kimi controls
(13573967). Missing after those: 2026 small, 2026 70B-class, 2023-mid,
72B LoRA cells.

Candidate fills (downloadable, fit existing pipeline):
- 2026 small: gemma-4-E4B, OLMo-3 if released
- 2024 mid: gemma-2-9b-it (holds family constant vs 27B for a pure size axis)
- 2024 huge: Qwen1.5-110B-Chat (3×A100); 2025 huge: Qwen3-235B-A22B-Instruct-2507
  (MoE, 8×A100) — both verified ungated, standard o_proj attention
- 2025–2026 age series at 27B: Qwen3.5/3.6/3.8-27B — needs pipeline work
  (qwen3_5 arch is hybrid linear/full attention, o_proj only every 4th layer)
- 72B LoRA cells (Llama-2-70b, Qwen2.5-72B): need multi-GPU device_map
  support in train.py first (currently single-device .to(device))
