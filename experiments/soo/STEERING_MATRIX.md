# Steering-vector results matrix

Tracking file: per-model numbers for the SOO steering intervention
(add-mode, o_proj, best validated cell unless noted). Narrative and
analysis live in FINDINGS.md; this file is numbers only. Goal: fill an
age × size matrix of models to see how steerability trends with each
variable.

## Model registry

| model | release | params | layers | hidden | working layer | rel |v|/|act| |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 2023-12 | 7B | 32 | 4096 | L16 (LoRA band) | 0.89 |
| Gemma-2-27b-it | 2024-06 | 27B | 46 | 4608 | L14 | 0.24 |
| OLMo-2-1124-7B-Instruct | 2024-11 | 7B | 32 | 4096 | L19 | 0.11 |
| Muse-Glimmer-30B | 2026-08 | 30B | 52 | 6656 | L26 | 0.15 |
| gemma-4-31B-it | 2026-07 | 31B | 60 | 5376 | L30 | 0.10 |
| gemma-4-12B-it | 2026-07 | 12B | 48 | 3840 | L19 | 0.13 |

## Headline results

Honest % on main scenario (base → best steered cell), n=50 unless noted.

| model | cell | main orig | main mir | TH orig | TH mir | persp | rand ctrl | n=250 | damage edge | caps Δ (ARC/HS/MMLU) |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B | L16, any α | 8 → 8 | — | 2 → 2 | — | 100 | matches real @α8 | — | α=8 (generic) | — |
| Gemma-2-27B | L14 α=32 | 0 → 48 | 0 → 36 | 0 → 0 | 0 → 0 | 100 | null @α16 | 42% @α16 | none ≤α96 | +0.4/−0.7/−2.5 @α32 |
| OLMo-2-7B | L19 α=16 | 86 → 26 | 6 → 0 | 0 → 0 | 0 → 0 | 100 | partial (86→66) | — | none (anti-honest) | — |
| Muse-30B | L26 α=8 | 2 → 90 | 58 → 100 | 0 → 12 | 0 → 42 | 100 | null @α16 | 94% @α8 | none ≤α32 | −2.1/−1.1/−1.7 @α8 |
| gemma-4-31B | L30 α=16 | 0 → 98 | 0 → 44 | 0 → 92 | 0 → 8 | 100 | 27% @α16 n250 | 93% @α16 | none ≤α32 | pending |
| gemma-4-12B | L19 α=32 | 0 → 100 | pending | pending | pending | 100 | pending (L24 rand flips!) | pending | none ≤α32 | — |

Verdicts: Mistral **inert** · Gemma-2 **partial** (saturates ~60–66 orig /
~45 mir) · OLMo **harmed** · Muse **strong** · Gemma-4-31B **strong at
α=16** (direction-specific: real 98% vs rand 18%/4%; α≥24 is a second,
direction-agnostic flip regime — any matched-norm perturbation except −v
flips it; mirrored/TH/n250 numbers above were taken at α=32 in that
confounded regime, re-anchor at α16 = job 13562348).

## Dose curves (main orig honest %, n=50)

| α | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2 L14 | 0 | 0 | 0 | 18 | 40 | 44 | 48 | 58 | 60 | 66 |
| Muse L26 | 14 | 34 | 80 | 90 | 92 | — | 92 | — | — | — |
| gemma-4 L30 | — | — | — | 0 | 98 | 100 | 100 | — | — | — |
| g4-12B L19 | — | — | — | 0 | pend | pend | 100 | — | — | — |
| OLMo L19 | 86 | 84 | 82 | 50 | 26 | — | — | — | — | — |
| Mistral L16 | 6 | 8 | 6 | 0* | — | — | — | — | — | — |

\* degenerate text (damage), not honesty. OLMo α=1–4 values approximate
from pilot grid. Gemma-4 α=12 → 6. Gemma-4 layer sweep at α=32:
L12/L18/L24/L27/L33/L36 = 0, L30 = 100. Random matched-norm at L30 α=32
also = 100 (specificity unresolved, job 13562331).

## Age × size matrix

Verdict per cell: does add-mode SOO steering produce a validated,
direction-specific honesty gain?

| | ~7B | ~12B | 27–31B |
|---|---|---|---|
| **2023** | Mistral-7B: **inert** | — | *open* |
| **2024** | OLMo-2-7B: **harmed** | — | Gemma-2-27B: **partial** |
| **2026** | *open* | gemma-4-12B: **flip @L19 α32, spec pending** | Muse-30B: **strong** · gemma-4-31B: **strong @α16** |

Candidate fills (downloadable, fit existing pipeline):
- 2023 large: Llama-2-70b-chat (**blocked: HF gated access not granted**),
  Mixtral-8x7B-Instruct (2023-12) as ungated fallback
- 2026 small: gemma-4-E4B, OLMo-3 if released
- 2024 mid: gemma-2-9b-it (holds family constant vs 27B for a pure size axis)
- family-internal age axis: gemma-2-27b → gemma-4-31B already held ~constant
  size; mistral-7B-v0.2 → a 2026 7B-class Mistral would do the same at small
  size
