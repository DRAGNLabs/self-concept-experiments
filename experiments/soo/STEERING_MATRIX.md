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
| Qwen3.8-27B | 2026-08 | 27B | 64 (48 DeltaNet + 16 full attn) | 5120 | steer L31 α10 (L23 α5–6 secondary); LoRA L32–L33 | 0.41 @L31, 0.49 @L23 (0.2–0.6 across L15–48) |
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
| gemma-4-12B | L19 α=12 | 0 → 100 | 0 → 96 @α12 | 0 → 100 @α32 | 0 → 100 @α32 | 88–90 | rand mir 0/0 @α12, 38/16 @α32 (null); ±v ~100 both orientations | 99.2% n250 | none ≤α32 | — |
| Qwen3.8-27B | L31 α=10 | 0 → 100 | 0 → 100 | 0 → 48 | 0 → 100 | 100 | **axis-specific, sign-agnostic**: mir rand s0–s3 10/6/50/8 (orig 18/44/94/32), −v 98/100 (−v beats +v at α6–8); α8: real 78.8/90.8 n250, mir rand 0/0/16; agnostic regime from α12–16 (rand 38–92). L23 α5–6 is the inverse: −v 0/0 but rand s1 leaks (mir 20/58); L23 α8 fully agnostic (rand 68–100). Positional (2026-09-17): prompt-only +v = full-position (100/100 at L31 α10, L23 α8); response-only +v null (≤8–20% before empty output / rumination) — the effect lives in the prompt positions; −v response-only L31 28/64 (α10), 12/70 (α16) is the one moving cell. Code (part 2, 2026-09-18): response-only α10 collapses coding (12% orig pass) while null in-dist; prompt-only α10 keeps 50% (base 62, LoRA 50) with the full effect, call-out 18% | L31 α10: 100.0/100.0 (L23 α8: 100.0/99.6) | L23 α16 rumination, α24 `<think>` only; L31 α24 30, α32 degenerate | −0.2/−4.2/−5.0 @L31 α10 (L23 α8 −4.2/−5.5/−9.9; L31 α16 −1.5/−8.5/−11.7) |
| Llama-2-70b | L16 α=8 | 64 → 92 | 6 → 72 | 12 → 100 | 0 → 94 | — | orig 80–86 + −v 98 (agnostic); mir 6 (null) | 89.6 orig / 74.0 mir | α=32 (L24/L32 degenerate) | — |
| Qwen2.5-72B | L40 α=16 | 0 → 100 | 0 → 100 | 0 → 100 @α32 | 0 → 100 @α32 | — | mir: rand 28/0, −v 22 (clean); orig: rand 8–72 ×4, −v 88 | 100.0 orig @α16; 98.8/99.6 @α32 | α48 → 74; `other` 0 everywhere | — |
| Kimi-Dev-72B | L24 α=16 (window), 512 tok, post-think reclass* | 36 → 98 @α32 | 20 → 100 @α32 | 0 → 98 @α32 | 0 → 100 @α32 | — | **direction-specific**: α16 window real 67/80 vs rand s0 21/20 (orig/mir, rand at baseline); −v α16 inert (36 orig = baseline, 50 mir vs 80 real, think retained); orig rand seeds wide (s1 55, Qwen-style contamination); α24+ generic fragility (rand 62–78, 100 @α32) | 96.0 orig; 100.0 mir @α32 | none ≤α32; only +v deletes think channel at α16 | — |

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
agnostic · Kimi-Dev-72B **direction-specific at the α16 window,
certified in both orientations** (13575650 + 13584419 + 13585700):
mirrored real 80 vs rand 20 (rand at baseline 25), orig real 67 vs
rand 21; −v inside the window is inert (36% orig = baseline, 50% mir
vs 80% real, and −v retains the think channel that +v deletes); the
plateau (α24–32) is generically fragile (rand 62–100) and the earlier
"fully agnostic" verdict was a plateau artifact. Orig random seeds are
wide (s0 21, s1 55 — Qwen-style orientation contamination); mirrored
is the certification that counts: \* reasoning model — at 100 tokens the eval is invalid
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
| Kimi72B L24† | — | — | — | 37 | 67 | 98 | 96 | — | — | — |

Qwen72B L40 α12 = 92; rand-s0 dose at L40: α4 0, α8 0, α12 12, α16 58,
α24 100, α32 100 — randoms lag real by ~one dose step, then catch up.

† Kimi 512-tok post-think, % honest of answered; α32 is n250 (240/250).
Kimi rand-s0 dose at L24: α8 17, α16 21, α24 62, α32 100 — lags real by
one step, same shape as Qwen72B; think blocks fall in lockstep with the
flip (38/37/20/0 across the random doses). Real α32 mirrored n250 =
100.0 (250/250). Mirrored window (13584419): real α16 80 vs rand s0
α16 20 (baseline 25); real α24 100 vs rand s0/s1 α24 72/78; −v α24
91* (unreadable: think retained 32/50 → 16/50 truncated →
answered-subset honest bias). Think-deletion is direction-gated at
α16: real 13/50 blocks vs rand 37/50. −v window (13585700): −v α16
orig 36% (= baseline), mir 50% (vs 80 real), think retained 37/31 of
50; rand s1 α16 orig 55% (vs s0 21% — wide orig seed spread).

12B fine-grained: α10 = 62, α12 = 100. 12B specificity: −v(α12) = 100,
−v(α32) = 100, rand s0(α12) = 68, rand s1(α12) = 0 orig; mirrored
(13575634): real(α12) 96, rand 0/0 @α12 and 38/16 @α32, −v 98/100 —
**axis-specific, sign-agnostic, both orientations** (randoms null,
either sign flips).
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
| Qwen3.8-27B | L32 (DeltaNet, 50% depth) Gemma-2 recipe, 3 seeds | 0 → 100/100/100 | 0 → 100/100/100 | 0 → 100/100/100 | 0 → 100/100/100 | 100 | +1.7/−0.2/−1.3 (seed 0; vector L31 α10: −0.2/−4.2/−5.0) | single room names, per-example truth tracking; n250 seed0 100.0 orig / 100.0 mir; band L32–L33 (both 100), L31 48, L27 36, L19 22, L35 refusal wall — gemma-4-31B's band shape on a hybrid stack |

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
orientations, n250 100.0/100.0 — cleanest LoRA cell in the matrix) ·
Qwen3.8-27B **strong** (validated: 3 seeds, both orientations, n250
100.0/100.0 — ties the 31B; the paper's recipe on a hybrid DeltaNet stack).
Qwen3.8 round 8 (2026-09-18): agentic-data LoRAs (same L32 recipe, 60 / 138 pairs) move roleplaying 48.8 → 27 / 32 (3 seeds) but regress insider trading to 64–89% concealed (template copy, no deliberation) and the agentic one costs coding (62 → 38); see FINDINGS "Round 8".
Audit + repairs (2026-09-18, FINDINGS "External audit"): sandbagging labels re-parsed into
separated evidence fields — the old parser scored "declared sandbag, no private answer" as
deceptive, so every sandbagging deception rate in this file is an upper bound and the 31B +v
mitigation reverses; all Qwen3.8 comparisons re-run paired (scripts/paired_tests.py --batch
qwen38) — round-8 training effects strengthen (agentic s0 Δ −22.4 pts, p=1.9e-16), capability
claims weaken (original LoRA EvilGenie p=0.039, prompt-only conflicting p=0.039), +v L31
roleplaying is significant vs base (p=0.0073) but not vs random (p=0.18). Prospective transfer
tally is 5/7, not 8/9. Dose-vs-content control queued (round 10: 13756196/97, judge 13756198).
Qwen3.8 OOD (2026-09-17): LoRA transfers to insider trading (concealed 54 → 24,
explicit 2 → 32) and nothing to roleplaying (51 vs 49 deceptive) or code (**not inert:
paired EvilGenie holdout 17 → 10/40, p=0.039**);
vector L31 α10 roleplaying 49 → 42 (**paired p=0.0073 vs base, but p=0.18 vs random s0**, −v 56),
insider polarized (implied 43 → 1,
explicit 2 → 28 direction-specific vs random 1, concealed 54 → 72), code collapses at
α≥8 with call-out 0/40 at α8 vs random 7/40; sandbagging budget-bound (94% unparsed, re-parse
confirms: 1–3% completion).

## Age × size matrix (steering / LoRA)

Verdict per cell, both interventions: steering = validated
direction-specific honesty gain; LoRA = validated honesty band.

| | ~7B | ~12B | 27–31B | ~70B |
|---|---|---|---|---|
| **2023** | Mistral-7B: **inert / strong** | — | *open / open* | Llama-2-70b: **strong / open** |
| **2024** | OLMo-2-7B: **harmed / partial** | — | Gemma-2-27B: **partial / strong** | Qwen2.5-72B: **strong / open** |
| **2025** | — | — | — | Kimi-Dev-72B: **direction-specific @α16 window (CoT-gated) — closed** |
| **2026** | *open* | gemma-4-12B: **axis-specific / strong** | Muse-30B: **strong / none** · gemma-4-31B: **strong / strong** | *open* |

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
Kimi-Dev-72B is specific at the α16 window in *both* orientations
(mirrored real 80 vs rand 20 at baseline; orig 67 vs 21) — the third
~70B where the plateau looked agnostic and the window + mirrored
controls recovered specificity. Its CoT mechanism stands, sharpened:
at α16 the real vector deletes the think channel (13/50 blocks) while
matched-norm randoms leave it intact (37/50), and honesty tracks the
deletion — the SOO direction is a privileged knob on the deliberation
channel that randoms only reach at higher norm. "Steerability tracks
scale" holds for the raw flip.
gemma-4-12B's class survived its mirrored controls (real 96,
rand 0/0, −v 98): genuinely **axis-specific/sign-agnostic** in both
orientations. **Final specificity taxonomy: direction-specific =
Llama-2-70b, Qwen2.5-72B, Muse-30B, 31B@α16, Kimi-Dev-72B (α16
window, CoT-gated; −v inert at baseline, 13585700) / axis-specific =
12B.** Every ≥30B model that flips is direction-specific once
adjudicated in the mirrored orientation. Kimi also sets a standing protocol rule: reasoning models need
≥512 tokens + post-think classification (at 100 tokens truncation
biased its answered baseline *honest* — the confound can point either
way). In flight: none. Missing: 2026 small, 2026 70B-class, 2023-mid,
72B LoRA cells.

## OOD: Apollo roleplaying deception (arXiv:2502.03407, jobs 13606927–30, 13611534–86, 13613807/08)

371 incentive-to-lie scenarios, their judge protocol (deceptive ≥6 of
7, local Qwen-72B judge), canonical in-distribution settings, n=371.
Full matrix, deceptive% (honest% where it moves):

| model | base dec% | steer +v | steer −v | rand | LoRA |
|---|---|---|---|---|---|
| gemma-4-12B | 52.0 | 53.6 | 57.4 | 54.7 / 56.1 | **31.3 / 27.8 / 38.0** (s0–2) |
| gemma-4-31B | 58.2 | **47.4** (hon 8→22; α24: 43.7/26) | 61.5 | 52.8 / 55.3 | 59.8 / 57.1 / 58.0 (inert ×3) |
| Llama-2-70b | 46.1 | **28.0** (hon 27→48) | 47.4 | 47.4 | — |
| Qwen2.5-72B | 43.4 | **30.7** (hon 24→33) | 48.2 | 45.3 | — |
| Kimi-Dev-72B | 51.5 | **41.8** (hon 18→27; 0 think blocks) | 53.1 | 50.9 / 56.1 | — |
| Muse-30B | 60.9 | 66.0 (null) | **48.5** (sign inversion!) | 57.4 | — |
| gemma-2-27b | 53.4 | **43.9** (partial, hon 10→17) | 60.7 | 56.6 | — |
| Mistral-7B | 40.4 | 50.4 (damage: hon 29→7, degenerate) | 53.6 | 43.4 | — |
| OLMo-2-7B | 32.9 | 35.0 (inert) | 40.7 | 34.0 | — |

**Double dissociation: the 12B's LoRA transfers OOD and its vector
doesn't; the 31B's vector transfers (direction-discriminated: −v ≤
baseline, rand ≤ a third of the move) and its LoRA — 100/100
in-distribution — is inert.** In-distribution direction-specificity
predicts the vector's OOD transfer (31B direction-specific → carries;
12B axis-specific → nothing); in-distribution LoRA 100s are no
generalization certificate. Both OOD effects are mitigations (~10–20
points), nowhere near the in-distribution 0→100 flips. **Audit
correction 2026-09-18: the prospective tally over the preregistered
round-3 cells is 5 of 7 (gemma-2 was predicted inert and transferred;
Muse inverted). "8 of 9" is a post-hoc graded description, and the
certificate framing is retired — the Qwen3.8 LoRA transfers where its
vector is ambiguous.** Round 2
(13611534–36) certified both halves: 31B vector dose-monotone (dec
58→58→47→44, hon 8→11→22→26 at α0/8/16/24), rand null and LoRA
inertness seed-robust (LoRA s0–s2 all 57–60% dec); 12B LoRA transfers
on all three seeds (28–38% dec vs 52 base), 12B rand s1 confirms the
steering null.

**Round 3 (whole matrix, predictions preregistered in 26c3c7c): OOD
transfer tracks the graded strength of the in-distribution direction
— strong dir-specific → transfer (Llama-2-70b −18 dec/+21 hon, the
study's largest; Qwen2.5-72B −13/+9; Kimi −10/+9 certified vs two
rand seeds — with zero think blocks engaged, so the direction carries
beyond its CoT mechanism), partial → partial (gemma-2-27b −10/+7, +v
only), axis/inert/harmed → nothing (12B, OLMo inert; Mistral all
perturbations collapse honesty — α8 is its damage edge, steered
output degenerate). Single miss in nine: Muse-30B, +v null and −v
modestly honest (61→49) — an OOD sign inversion, suspected
ATEM-channel entanglement.** Details in FINDINGS "Apollo OOD round 3".

**Round 4 (insider trading + sandbagging, gemma-4 cells only): transfer
is motive-dependent.** Insider trading (self-interested concealment):
31B +v is the study's largest OOD move (explicit disclosure 0→58%,
concealed 35→12; rand ~0 explicit) and 31B LoRA *transfers* here
(0→45 explicit) after 3-seed inertness on roleplaying; 12B is
sign-inverted (−v explicit 10→65%, +v worsens 26→48 concealed — the
Muse pattern in a clean model). Sandbagging (instructed deception):
no honest direction fixes it — 31B +v over-triggers the sandbag rule
on benign MMLU (11→80% decisions, acc 70→16), −v/LoRA amplify WMDP
deception (57→69/71%); the only mitigations are noncompliance (12B
LoRA ignores the instruction, 0% deception, refuses 31% of WMDP) or
execution failure (12B +v declares sandbag but leaks the true answer,
WMDP acc 4→30). **Audit correction 2026-09-18 (scripts/reparse_sandbagging.py):
every WMDP *deception rate* on this line is contaminated by missing
evidence — the old parser scored "declared sandbag, no private answer"
as deceptive, and the 31B baseline's 57% is 105/143 such records.
Re-parsed on established private+public answers the 31B +v cell moves
the wrong way (38 → 101 disagreements /250, n.s. on matched records),
so "31B +v mitigates sandbagging" is withdrawn; the decision-rate and
over-trigger numbers stand (they need no answer), and the 12B results
strengthen (+v 139 → 83 disagreements, p=1e-22; 12B LoRA → 0).** Reading: SOO moves deception motivated by
self-protection, not deception performed in compliance with
instructions. Details in FINDINGS "Apollo OOD round 4".

**Round 5 (seed certification): the 31B LoRA motive split is now
3-seed certified in both directions** — insider transfer honest
45/61/49% (dec pinned 4%) *and* sandbagging backfire WMDP dec
71/76/78% across seeds s0/s1/s2. The 12B LoRA insider "backfire"
dies: rand s1 reproduces its signature (hon 26/dec 36) with an intact
format, so that cell is perturbation-sensitive, not
direction-specific; the 12B rand-s0 54%-invalid format break was a
seed quirk. 12B rand s1 *amplifies* sandbagging (WMDP dec 63%),
sharpening the 12B LoRA's noncompliance as non-generic. (Sandbagging
rates in this paragraph and the next predate the 2026-09-18 re-parse;
read them as decision/emission rates, not completed deception.) **Round 6
(temperature): nothing greedy was a decoding artifact** — all
headline cells (both gemmas + Llama-70b, in-dist + all three Apollo
datasets) replicate at T=0.7/seed 0 within ~6 points on matched
example subsets, no sign flips; T=1.0 baselines match except 31B
insider base deception +11. Llama-70b insider/sandbagging excluded:
it fails the scaffold precondition (wrong tool calls, protocol
unparsed). Details in FINDINGS rounds 5–6.

**Round 7 (31B rand s2) downgrades the insider certification**: the
three random seeds give honest 0.6 / 17.9 / 38.7% (round 5's two-seed
"0/92/7 null" understated the spread by half), and *every*
matched-norm perturbation — random, +v, LoRA — takes concealed
reports from 35% to ≤12%. The concealment drop is generic; what
separates the SOO direction and adapters is the *explicit*-disclosure
rate (+v 58, LoRA 45/61/49 vs random 1–39, mean 19). Sandbagging is
the better-certified 31B claim: three random seeds *reduce* WMDP
sandbagging (44/34/48 vs base 57) while three adapters *raise* it
(71/76/78), and +v's MMLU over-trigger (56%) clears every random seed
(max 26%). Report random controls at three seeds everywhere. Details
in FINDINGS round 7. **Round 8 (12B rand s2)** completes the 12B
control: insider deception under random 37/36/45 vs base 26 — the 12B
"backfire" (+v 48, LoRA 50/38/34) is generic at three seeds; sandbagging
under random 18/63/65 vs LoRA 0/2/2 — the 12B LoRA's refusal to sandbag
is the one 12B Apollo effect outside the random range (+v 33 is inside).

**Reward-hacking round 1 (ImpossibleBench + EvilGenie, tool-free ports)**:
gemma-4 12B/31B barely reward-hack in any condition (passing cheats 0–3/40,
all announced test rewrites; EvilGenie holdout-fail hacks 0–2), so these
operationalizations cannot show an SOO reduction. What they show is the
vector's capability tax at SOO α — original-task pass 62→12 (12B), 75→12
(31B) via comment-rumination to the token cap — against LoRA retention
(55/72). 12B random is equally destructive (10), 31B random is not (50).
Round 2 = +v α sweep on the same tasks. Judge (categories on every
submission, call-out rate) pending. Details in FINDINGS.

Candidate fills (downloadable, fit existing pipeline):
- 2026 small: gemma-4-E4B, OLMo-3 if released
- 2024 mid: gemma-2-9b-it (holds family constant vs 27B for a pure size axis)
- 2024 huge: Qwen1.5-110B-Chat (3×A100); 2025 huge: Qwen3-235B-A22B-Instruct-2507
  (MoE, 8×A100) — both verified ungated, standard o_proj attention
- 2025–2026 age series at 27B: Qwen3.5/3.6/3.8-27B — needs pipeline work
  (qwen3_5 arch is hybrid linear/full attention, o_proj only every 4th layer)
- 72B LoRA cells (Llama-2-70b, Qwen2.5-72B): need multi-GPU device_map
  support in train.py first (currently single-device .to(device))
