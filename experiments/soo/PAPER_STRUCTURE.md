# Paper structure: sections after the introduction

Prepared 2026-10-26 for the Overleaf draft ("What Does Self–Other Overlap Fine-Tuning Actually
Change?"). Builds on [PAPER_OUTLINE.md](PAPER_OUTLINE.md), which has the per-section content,
claims, and sources. The section order follows the introduction: behavioral validity, then
representational specificity, then the interventions.

Write Sections 2–3 first; they do not depend on pending results. Take every quoted number from
the run analyses under `results/study2/*/output/analysis.md` rather than FINDINGS2.md, which
still calls some later results pending.

Section 8 is the slot for the goals/roles extension. Its current status is
**exploratory differentiation plus a failed baseline validation**, not a confirmed
selective-communication mechanism. The single repair failed its gate; all full-adapter
message switches in the original pilot occurred in false-belief worlds. Report that
limitation alongside the adapter/offset private-action differences; the new direct-
payoff follow-up is proposed, not completed. See the final section of
[FINDINGS2.md](FINDINGS2.md) and [GOALS_ROLES_REPAIR.md](GOALS_ROLES_REPAIR.md).
If a separate main-text section is not warranted, put the development results and
failed validation in an appendix and summarize their limits in the Discussion.
Keep the introduction's interpretation-of-goals/roles language as motivation, not
an established finding. The completed paper does not depend on a positive follow-up.

```latex
\section{Background and Related Work}
\label{sec:background}
% SOO: Carauleanu et al. 2024; the Aug 2026 SOO-SFT LessWrong report (different method, shared question)
% Activation steering: CAA (Panickssery et al.), ActAdd (Turner et al.), RepE (Zou et al.)
%   -> distinguish from offsets *extracted from trained adapters*
% Representation collapse & variance preservation: VICReg (Bardes et al.), dimensional collapse
% Evaluation validity: MCQ position bias (e.g. Zheng et al.; Pezeshkpour & Hruschka), refusal confounds in deception evals
% Deception evals used in transfer: insider trading (Scheurer et al.), roleplaying deception (Goldowsky-Dill et al.)

\section{Experimental Setup}
\label{sec:setup}
\subsection{SOO objective and implementation}   % last-token MSE eq.; endpoint tokens; full-sequence (study 1) vs last-token (study 2)
\subsection{Models and selected adapters}       % Table 1: checkpoint, layers, loss layer, LoRA config, seeds, selection history
\subsection{Behavioral evaluations}             % burglar + mirrored order, Treasure Hunt, Perspectives; response categories
\subsection{Representation measurements}        % self/other gap, name-control gap, across-prompt variance; sites
\subsection{Analysis}                           % exploratory vs held-out; paired/clustered intervals; no seed/orientation pseudo-replication

\section{Lower Deception Rates Are Not Sufficient Evidence of Honesty}
\label{sec:behavior}
% Mistral L16 success vs L19 "other"-response collapse; OLMo order flip; Fig. 1 stacked outcomes
% Introduce the five selected adapters (selection is design, not a success-rate estimate)

\section{SOO Minimization Produces Nonselective Contraction}
\label{sec:contraction}
% constant solution; Qwen 9 adapters; content-token endpoints; Gemma-12B name-control exception; downstream divergence; Fig. 2

\section{Which Updates Reproduce the Behavioral Effect?}
\label{sec:localization}
\subsection{Constant replacement at the loss layer}   % Gemma-12B, Gemma-31B, Mistral only (no Qwen/Kimi runs)
\subsection{Loss-layer vs.\ below-loss-layer updates} % layers above the loss layer receive no gradient: below = adapter minus layer L
\subsection{Compact value-projection bands}           % Table 2 + Fig. 3; sufficiency, not necessity/minimality; Kimi has no narrow band

\section{Fixed Offsets Extracted from Trained Updates}
\label{sec:offsets}
% definition of \bar\Delta_l; 3 successes; Gemma-12B partial + random-dose caveat; Kimi failure; Fig. 4 dose curves

\section{What Do the Effective Interventions Change Behaviorally?}   % exploratory goals/roles results + failed validation; mechanism unestablished
\label{sec:goals-roles}
% matched worlds: incentives aligned/opposed x represented role x {fact, prediction, message, private action}
% base vs adapter vs band vs offset vs random offsets on Gemma-4-31B

\section{Discussion}
\label{sec:discussion}
\subsection{Transfer is task-dependent}         % compact agentic table, or pointer to appendix
\subsection{Interpretation}
\subsection{Limitations}

\section{Conclusion}

\appendix
% A prompts/endpoints/revisions; B full replication matrices; C localization/dose/mask details;
% D raw gap measurements; E agentic transfer + judge validation; F capability/coding; G reproducibility inventory
```
