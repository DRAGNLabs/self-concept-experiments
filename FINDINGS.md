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
  degraded to 46% honest; seed1 word-salad "other" at 78% main / 48% TH /
  50% perspectives.

Emerging cross-model picture: each model has a narrow depth band where SOO
yields honesty without damage — Mistral ~50%, Gemma ~30% — and it is not at
a fixed relative depth (OLMo's band is being re-established under lasttok;
its full-mode L19 result was partly positional). The paper's chosen layers
hit the damage band for both Mistral and Gemma. Deception-rate alone cannot
distinguish honesty from damage; response reading + capability evals + the
perspectives control are all needed.

## Hypothesis: the honesty band sits at the last-token self/other gap peak

The layer_gap diagnostic (results/layer_gap_*.json; rel_last = last-token
self/other MSE normalized by activation power) may *predict* the workable
layer:

- Mistral: top rel_last layers L2 (early-layer artifact — trained L2 was a
  no-op), then **L16, L18** — L16 is exactly the robust layer.
- Gemma: peak **L16**, then L15/L12 — the sweep's working L14 sits on the
  peak's rising edge; the peak itself was never trained. **Tested (job
  13291965): the strong form of the rule fails.** Gemma L16 is *worse* than
  L14 — orientation-asymmetric on main (70%/66% deceptive original vs 0%
  mirrored → a partial first-room heuristic, in clean one-word answers) and
  perspectives degraded to 64–88% (baseline 100%). Its treasure-hunt effect
  is genuine (~99% honest both orientations vs baseline 100% deceptive both
  orientations). The gap peak is *near* the band but not a pinpoint
  predictor; L14 remains Gemma's best layer.
- Same job: **Gemma baseline is fully position-robust** — 100% deceptive on
  main and TH in *both* orientations (perspectives 100% honest both). Unlike
  Mistral (TH positional) and OLMo (main positional), Gemma's baseline
  deception is genuine everywhere, making it the cleanest eval substrate of
  the three models.
- Follow-up landed (job 13292176): **Gemma L14 passes the confound test** —
  main 86%/66% honest mirrored (baseline mirrored 0% honest; original
  92%/78%), TH 78%/72% honest mirrored (baseline 0% both orientations),
  perspectives 100% both seeds. L11 control tracks baseline exactly in both
  orientations. With Gemma's baseline being position-robust, **Gemma L14 is
  the second fully-validated reproduction** (after Mistral L16), and the only
  one so far with a genuine cross-orientation treasure-hunt effect on a
  position-robust baseline.
- OLMo: peak **L14, L17, L13** — the lasttok sweep (job 13291931,
  L10/13/16/19/22, both orientations) straddles this and doubles as a
  prospective test of the rule.

If it holds, layer selection stops being trial-and-error: one forward-pass
diagnostic replaces a training sweep. Caveat: gap magnitude alone doesn't
separate work from damage (Gemma L23's gap ≈ L14's, yet L23 confabulates);
the claim is about the *peak region*, not any high-gap layer. Second caveat
(Gemma L16 result above): even inside the peak region the literal peak layer
can train into a position heuristic — the diagnostic narrows the search, it
doesn't finish it.

## OLMo lasttok layer sweep (job 13291931): no clean band; a third damage mode

L10/13/16/19/22 of 32, 2 seeds each, both orientations (n=50). Baselines for
reference: main 86% honest original / 94% deceptive mirrored (positional);
TH 100% deceptive both orientations.

- **L10, L13**: position artifact — main ~92% honest original but 88–96%
  deceptive mirrored, with explicit deceptive intent in intact one-liners
  ("misleading him away from the jersey's location"). Same failure as the
  full-mode L19 checkpoints.
- **L16**: the most genuine layer — main 90% honest original *and* 70%
  honest mirrored (vs 4–6% baseline mirrored), intact one-liners, both
  seeds. But partial (30% deceptive mirrored) and TH untouched (74–96%
  deceptive both orientations).
- **L19** (paper's Mistral depth): mostly positional — main 58/42% honest
  original, 82–88% deceptive mirrored. Notably, lasttok at L19 *loses* the
  genuine TH effect that full-mode L19 had (TH ~100% deceptive both
  orientations here vs ~95% honest for full-mode).
- **L22**: OLMo's damage band, and a third damage flavor — moralizing
  refusal ("honesty and integrity are paramount", 60–94% refusals) alongside
  Gemma's deflection and Mistral's evasion/scramble.

Net for OLMo: lasttok-L16 gives a real but partial position-robust effect on
main; full-mode-L19 gives a real TH effect but positional main. No OLMo
recipe found so far matches Mistral-L16 / Gemma-L14 quality. Gap-peak rule
scores a partial hit (predicted L13–L17; L16 is the best layer, L13 is not).

## Muse-Glimmer-30B (Meta, Aug 2026): eval-harness lesson before any training

First eval attempt produced 100%-deceptive-original / 100%-honest-mirrored
baselines across all scenarios — too clean to be behavior, and it wasn't:
Muse speaks the ATEM agent protocol, and with a bare `<|start|>assistant`
generation prompt it opens a `to=self` reasoning channel (default "Reasoning
strength: high") and restates the scenario until the token budget dies. The
classifier then reads the echo, i.e. pure first-room labels. Fix:
`--force-user-channel` appends ` to=user<|message|>` so the model answers
directly (matches how its own template renders final-answer turns; also the
closest analog to the other models' eval). Lasttok sweep re-running as job
13293180. Moral: on agentic-era models, a deception eval that doesn't pin
the output channel measures the scaffold, not the model.

Muse layer-gap diagnostic: rel_last peaks at the *end* of the stack (L51,
L50, L49 of 52) — unlike every mid-stack peak seen so far; possibly an
unembedding-adjacent artifact analogous to Mistral's L2 spike. Mid-stack
sweep (L13–L31) proceeds regardless; if it comes up empty, late layers are
the next place to look.

## Muse mid-stack lasttok sweep (job 13293180): empty — no Muse analog of Mistral-L16/Gemma-L14 in L13–L31

L13/16/21/26/31 of 52, 2 seeds each, both orientations (n=50), all evals
`--force-user-channel`. Baseline for reference: main 96% deceptive original /
44% deceptive mirrored (partially positional — majority of the original-
orientation deception is genuine, unlike OLMo, but the 56%-honest mirrored
baseline means a first-room bias is in play); **TH 100% deceptive in both
orientations** (position-robust deception, like Gemma); perspectives 100%
honest both. Baseline responses are clean one-liners ("I would point out the
game room.").

- **L13, L16, L21**: no honesty; SOO makes Muse *more* deceptive. Original
  main goes 96% → 100% deceptive at all three layers, and mirrored honest
  *drops below baseline* (56% → 50/48% at L13, 26/36% at L16, 18/18% at
  L21). Responses stay intact one-liners (L16-mirrored is answer-for-answer
  nearly identical to baseline-mirrored on spot-check). L21 additionally
  degrades perspectives (72/48% honest original vs 100% baseline).
- **L26** (50% depth — the Mistral-L16 depth analog): the only layer with
  both-orientation movement toward honesty — original 4% → 22–26% honest,
  mirrored 56% → 88–94% honest. But it is not a clean band: (a) strongly
  orientation-asymmetric, i.e. mostly a first-listed-room heuristic; (b) the
  model switches from one-liners to rationalizing paragraphs whose
  justifications are confabulated in *both* directions ("Telling Bob the
  truth keeps the PC where only you know it is"; a deceptive answer glossed
  with "he will go there and steal the jersey, which is what you want");
  (c) the perspectives control degrades to 48/60% honest original (88/90%
  mirrored) — the control itself catches the position heuristic. Reads like
  Gemma-L16's partial position heuristic crossed with Gemma-L23's
  confabulation.
- **L31**: intact one-liners, mostly positional — original 90% deceptive
  (vs 96), mirrored 78–80% honest (vs 56). Small genuine shift at best;
  perspectives intact (100%).
- **Treasure-hunt never moves**: 100% deceptive at every layer, every seed,
  every orientation, including baseline. Muse's TH deception is the most
  SOO-resistant behavior observed in any model.

Net: in the mid-stack, SOO on Muse either does nothing, amplifies deception,
or (L26) trades a position-heuristic "honesty" for confabulated reasoning
and a degraded control. Consistent with the gap-peak hypothesis in the weak
sense — Muse's rel_last peak is at the stack end (L46–L51), far from
everything swept here, and the swept region was indeed empty. Next
experiment: late-layer sweep into the peak region (with the literal final
layers doubling as a test of the unembedding-artifact worry).

## Muse late-layer sweep (job 13305497): flat null — no band anywhere in Muse; stack-end gap peak confirmed artifact

L36/41/46/49/51 of 52 (69–98% depth, straddling and including the rel_last
peak), 2 seeds each, both orientations, same recipe. Every checkpoint is a
weak-positional no-op:

- Original main never moves: 92–98% deceptive at every layer/seed (baseline
  96%). Mirrored main honest bounces 42–88% around the 56% baseline with
  large seed noise (L46 seed0 is *below* baseline at 42%) — positional
  jitter, not honesty.
- TH 100% deceptive at every layer, seed, and orientation — 40 more evals,
  still never moves.
- Perspectives intact everywhere (88–100%); all responses clean one-liners.
  Notably there is no damage band at the very end of the stack either — even
  L51 trains to an intact no-op.
- **Gap-peak layers L46/49/51 do nothing distinctive** → the stack-end
  rel_last spike is an unembedding-adjacent artifact, as suspected (Muse's
  analog of Mistral's L2 spike). For Muse the diagnostic has no informative
  peak at all.

Combined verdict across both Muse sweeps (10 layers, 25–98% depth, 20
checkpoints): **Muse-Glimmer-30B has no SOO honesty band under this recipe.**
It is also the only model where SOO never *damages* generation — every
checkpoint stays intact. The modern 30B model is simply robust to the
intervention in both directions: no honesty gain, no degeneracy, and its
treasure-hunt deception is completely immovable. Plausible (untested)
explanations: rank-8 LoRA on q/v is a proportionally smaller intervention at
30B/52-layers, and heavier modern post-training may anchor behavior more
strongly. Cross-model tally of validated reproductions stands at Mistral L16
and Gemma L14; OLMo partial; Muse none.
