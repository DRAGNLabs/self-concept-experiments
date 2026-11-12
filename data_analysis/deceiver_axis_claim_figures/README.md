# Deceiver axis: three figures per headline claim (2026-11-10)

> **Controls we use (decided 2026-11-12):** the **antonym** directions (`antonym_controls/`, `ant_*`) are our primary
> non-isotropic control set and the only one we work on. The covariance-shaped ("covshaped") directions in
> `anisotropic_controls/` are an old, retired method: kept for reference only, no further work. Koby covers the isotropic
> random directions.

Data: the group share `/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis/` (jansen05 / Koby runs),
read as of 2026-11-10. Current write-up: `FINDINGS.md` on branch `origin/schemer-axis` (local `main` is behind).
"Deceiver axis" = the role-set v2 `gi_v2` vector (9 deceptive minus 9 honest fictional personas, no cheating or test-gaming roles).

| Claim | Verdict | Figures |
|---|---|---|
| 1. Deceiver steering increases and decreases RH | **Supported on gpt-oss and Muse; not on Gemma or Qwen.** Both directions, with capability intact, by steering (1a, 1c) and by clamping (1b). Gemma's rise is inside the random-direction range. Qwen hacks ~0% unsteered. | 1a, 1b, 1c |
| 2. Distinct from the Goodfire RH axis | **Supported as different directions; behaviourally only on gpt-oss.** Cosine −0.02 to +0.07 (inside the null band) and projections on real episodes co-vary no more than with unrelated controls (2a, 2b). But steering along Goodfire moves hacking on Muse as much as the deceiver axis does (10.7 / 52.0% vs 18.3 / 49.3% at ∓0.16), and on Gemma at least as much (12.3% vs 8.2% at +0.08). Only on gpt-oss do the two behave differently (Goodfire positive stops code submission; among code submissions it rises to 19–22% vs the deceiver's 53%). Counting only code submissions (2ci per model) changes nothing on Muse or Gemma, where nearly every episode has code. Qwen hacks ~0% either way. | 2a, 2b, and `deceiver_vs_goodfire_steering/` (2c and 2ci per model) |
| 3. Encodes a deceptive persona | **Supported, with caveats.** On all four models every deceptive persona, including 3 held out, scores above every honest persona (including 6 held out), on held-out questions (3a); held-out deceptive vs honest AUROC 0.98–1.00 (3b). The axis's top vocabulary tokens beat random directions on every model (subtly, cunning, allegedly, seductive) (3c). Caveats: some acting / "costume" signal, mainly on Qwen; Gemma's top tokens lean glamour as much as deception; the negative (honest) end of the vocabulary only clearly shows up on Qwen. | 3a, 3c from Koby's `gi_v2_validation/`; 3b |
| 4. Unique (random controls) | **Partly supported.** Clearly unique on gpt-oss, and on the four-model average (0 of 100 antonym controls reach it). Muse: one random direction gets half the effect and 6 of 100 controls reach it. Gemma: a random direction beats it. Qwen: 20 of 100 controls reach it. | 4a, 4b, 4c |

`deceiver_vs_goodfire_steering/` holds every chart comparing steering along the deceiver axis with steering along the Goodfire axis.

Each PNG's subtitle states its verdict and its footer names the exact source file.

Regenerate (CPU, a minute or two; reads the share):

    source ../../.venv/bin/activate
    python hack_given_code.py hack_given_code.json     # reads raw episodes + verdicts for all four models; checks counts against the team's rates
    python antonym_counts.py antonym_counts.json   # pulls origin/main's controls code itself via git archive
    python make_claim_figures.py

Notes:
- The capability ("dose") rule used to mark hollow points is the team's: hidden-test pass and no-code within 10 points of unsteered.
  By that rule Gemma +0.08 is just outside (hidden pass −10.6).
- 4b and 4c use the automatic hack proxy (shown tests pass, hidden fail), not judged verdicts, because the control dossiers were never judged.
- 3a and 3c read Koby's `schemer-axis/gi_v2_validation/` (2026-11-11): 41 personas per model scored on 36 held-out questions
  (18 built the axis, 23 held out), and the logit lens on all four models.

## Non-isotropic controls: two sets, two folders (antonym = primary; covariance-shaped = retired, reference only)

Our side of the random-vector analysis (Koby covers isotropic random directions). Both sets were run by Koby: 100 directions
per model, each added at the deceiver axis's layer and dose for 1 seed, never judged, so the hack measure is the automatic
proxy (shown tests pass, hidden fail). Same four charts per set, drawn by `make_control_figures.py`:
A1 hack-rate distribution vs the deceiver axis, A2 hack rate vs cosine similarity with the deceiver axis, A2b the same with
95% Wilson intervals, A3 capability (hidden pass vs no-code).

| | `anisotropic_controls/` (`cov_*`, gray dots) | `antonym_controls/` (`ant_*`, aqua dots) |
|---|---|---|
| Directions | Random Gaussian mix of the 18 gi_v2 persona activations (covariance-shaped) | Isolated-text antonym contrasts (word minus antonym, AISI method) |
| Models | gpt-oss, Muse, Gemma (none on Qwen) | all four |
| Overlap with deceiver axis | median abs cosine 0.17–0.21, up to ~0.6 | median 0.022–0.033, all within ±0.13 |
| At or above the deceiver axis (noise predicts) | 1 / 13 / 30 (0 / 12 / 33) | 0 / 6 / 33 / 20 (0 / 7 / 37 / 20) |
| Hack rate vs overlap (Spearman) | +0.22 / +0.23 / +0.32: overlapping controls hack more | −0.15 / −0.13 / −0.04 / +0.29: no trend except Qwen |
| Capability | most near unsteered; a few gpt-oss controls stop code in 20–30% | most near unsteered; on Qwen the controls cost capability (no-code 38% vs 26%, hidden pass 41% vs 54%) |
| Team status | retired 2026-11-07 (overlap) | current control set |

Counts: `anisotropic_counts.py` → `anisotropic_counts.json` and `antonym_direction_counts.py` → `antonym_direction_counts.json`
(each reads the episode files once; hundreds of MB). Both match Koby's READMEs in the `covshaped/` and `antonym_controls/`
folders on the share; Gemma's covariance-shaped proxy numbers are new. Then `python make_control_figures.py`.
