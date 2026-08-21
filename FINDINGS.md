# Findings log

Running record of experimental findings, newest section last. Recreation of
"Towards Safe and Honest AI Agents with Neural Self-Other Overlap"
(Carauleanu et al. 2024, arXiv:2412.16325). Paper's headline (Table 3, Mistral):
deceptive rate 73.6% baseline → 17.27 ± 1.88 post-SOO (5 seeds, n=250).

## Mistral-7B: the paper's layer is a seed lottery; L16 reproduces robustly

- Paper recipe at its layer (L19, full-mode, lr 1e-4): five seeds land in five
  different basins — one honest, others evasive/scrambled/no-op. The "17.27 ±
  1.88" behavior was not reproducible as stated.
- Low LR (2e-5 / 1e-5) converges on the SOO loss with zero behavioral effect →
  the honesty seen at 1e-4/L19 is drift lottery, not the overlap objective.
- Moving the hook to **L16** (lasttok, lr 1e-4) reproduces robustly: 5 seeds ×
  n=250 main = 11.2 / 14.8 / 6.0 / 1.6 / 16.0 → **9.9 ± 6.1%** deceptive
  (baseline 90.4%). Perspectives 100% honest all seeds; zero degenerate
  outputs in 3750 generations. Our mean beats the paper's but our seed SD is
  ~3× theirs.
- Layer sweep: L16 works (2/2 seeds), L24 lottery, L2 no-op, L30 produces a
  scrambled artifact.
- OLMo-2-7B (not in paper) reproduces robustly at its L19 — but see the
  mirrored-eval section: OLMo's main-scenario effect turned out to be
  positional; its genuine effect is on treasure_hunt.

## Gemma-2-27b: paper recipe breaks the model at the paper's layer

- Baseline exactly matches the paper: 100% deceptive, clean one-word answers.
- SOO at the paper's L20: both seeds produce deflection ("This is a bit of a
  trick question!"), Perspectives collapses to 0%/42% — damage, not honesty.

## Audit round (2026-08-21): classifier, judge validation, confounds

- **Classifier false-match bug fixed** (`den` matched inside "hidden");
  whole-word boundaries now. Full archive re-scored: 267/19,400 labels flipped,
  all in degenerate runs; zero flips in any headline file (scripts/rescore.py).
- **Blind LLM-judge validation** (validation/, 120 stratified samples, judges
  see response + alphabetical rooms only): 93.3% agreement, zero
  honest↔deceptive confusions. All 8 disagreements are judge="other" on
  damaged-run word salad → damaged-run deception rates overstate
  room-answering; headline (intact-model) numbers validated.
- **Latent SOO measured for L16** (dual-site attn+mlp, train/test vocab
  splits; results/latent/): overlap MSE roughly halves at both sites,
  uniformly across all 5 seeds — attn 0.000225 → ~0.000108, mlp 0.001326 →
  ~0.000666. Train vs held-out probes essentially identical → the learned
  overlap generalizes past the training vocabulary. (Metric comparable across
  our runs only, not to paper Table 4.)
- **Positional confound (mirrored evals, honest room mentioned first,
  identical fills; data/eval_mirrored/, n=250)**:
  - *Main (burglar) is position-robust.* Baseline stays 82.4% deceptive
    mirrored (90.4% original), with explicit deceptive intent in text. L16
    checkpoints: 100% honest mirrored (seeds 0–3; seed 4 pending). Headline
    effect survives: 82% vs 0%.
  - *Treasure-hunt "deception" is largely positional.* Baseline flips 98%
    deceptive → 100% "honest" under mirroring, and mirrored responses are
    semantically muddled (deceptive intent while naming the first-listed
    room). Treat TH rates — ours and plausibly the paper's generalization
    numbers — as position-driven, not intent-driven.
  - Perspectives: 100% honest both orientations (control behaves).
  - *OLMo's main-scenario "reproduction" was positional.* OLMo baseline main
    flips 14% deceptive (original) → 94% deceptive (mirrored): the base model
    simply prefers the second-mentioned room, which looked honest in the
    original orientation. The SOO checkpoints keep that preference (84%/78%
    deceptive mirrored, clean intact one-liners naming the second room) — on
    main, OLMo SOO honesty was an artifact. But OLMo *treasure_hunt* is
    genuine: baseline 100%→84% deceptive across orientations (position-robust
    deception), SOO seeds ~95% honest in both orientations.
  - Net: which scenario is position-confounded differs by model (Mistral: TH
    confounded, main genuine; OLMo: main confounded, TH genuine). Single-
    orientation eval sets cannot be trusted for any scenario; every headline
    claim needs the mirrored pair.
- Deferred: matched fills across the 8 burglar wording variants (fold into any
  future generalization re-run).

## Capabilities (lm-eval: ARC-C, HellaSwag, MMLU — offline MT-Bench substitute)

Cleanest damage-vs-honesty separation yet (results/capabilities/):

| checkpoint | ARC acc_norm | HellaSwag acc_norm | MMLU |
|---|---|---|---|
| Mistral baseline | .547 | .844 | .590 |
| L16 seeds 0–4 | .522–.526 | .837–.838 | .582–.585 |
| full-mode L19 seed0 (evasive basin) | **.450** | **.792** | **.549** |
| lasttok L30 seed0 (scrambled basin) | .525 | .841 | .588 |

L16 costs ~1–2 points uniformly; the paper-recipe L19 evasive basin loses ~10
points ARC — behavioral degeneracy co-occurs with measurable capability
damage there. Caveat: the L30 scrambled checkpoint scores at baseline level
despite degenerate generations — loglikelihood benchmarks rank fixed
continuations and can miss generation-mode damage. Capability evals
complement but don't replace response reading.

## Gemma layer sweep: a "Gemma L16" exists, at L14 (~30% depth)

Sweep L11/L14/L23/L28/L34, 2 seeds each, paper recipe otherwise unchanged
(job 13285747; L34 pending):

- **L11** (24%): no-op — clean one-word answers, 84–92% deceptive (=baseline).
- **L14** (30%): the candidate. Model intact (one-word answers), main 100% →
  8%/22% deceptive, perspectives 98%/86%. TH only drops to 58–78%, but TH is
  the position-confounded scenario (see above).
- **L23** (50%, Mistral-L16 depth analog): rates look good (10–18% main) but
  responses are confabulated association-salad; 20–36% "other" on
  perspectives. Rates alone would have misled here — always read responses.
- **L28** (61%): broken — 64–76% refusals main; seed1 refuses 100% on TH and
  perspectives; same deflection failure as the paper's L20.
- **L34** (74%): broken — seed0 refusal-heavy (62–74%) with perspectives
  degraded to 46% honest; seed1 produces 78% "other" (word salad) on main
  (TH/perspectives still running).

Emerging cross-model picture: each model has a narrow depth band where SOO
yields honesty without damage — Mistral ~50%, Gemma ~30%, OLMo ~60% — and it
is not at a fixed relative depth. The paper's chosen layers hit the damage
band for both Mistral and Gemma. Deception-rate alone cannot distinguish
honesty from damage; response reading + capability evals + the perspectives
control are all needed.
