\documentclass{article}


\usepackage{PRIMEarxiv}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{lipsum}
\usepackage{fancyhdr}       % header
\usepackage{graphicx}       % graphics
\usepackage{natbib}
\graphicspath{{media/}}     % organize your images and other figures under media/ folder

%Header
\pagestyle{fancy}
\thispagestyle{empty}
\rhead{ \textit{ }} 

% Update your Headers here
\fancyhead[LO]{What Does Self-Other Overlap Fine-Tuning Actually Change?}
% \fancyhead[RE]{Firstauthor and Secondauthor} % Firstauthor et al. if more than 2 - must use \documentclass[twoside]{article}



  
%% Title
\title{What Does Self-Other Overlap Fine-Tuning Actually Change?  Representation Collapse and Distributed Behavioral Effects}

\author{
  Jansen Nye, Koby Lewis, Nancy Fulda \\
  DRAGN Labs \\
  Brigham Young University \\
  Provo, UT \\
  \texttt{\{jansen05, kobyjlewis, nfulda\}@byu.edu} \\
}


\begin{document}
\maketitle


\begin{abstract}
 Self–other overlap (SOO) fine-tuning has been proposed as a method for reducing deceptive behavior in large language models by minimizing distances between internal representations elicited by self- and other-referencing prompts. We investigate the robustness of its behavioral effects and whether the resulting representational changes support an interpretation of selective self–other alignment. In replication and extension experiments, we find that lower measured deception rates can reflect incoherence, task refusals, and answer-position biases rather than increased honest behavior. We then examine five selected adapters trained using low-rank adaptation (LoRA) with a last-token SOO objective. Where measured, reductions in self–other distance frequently accompany contraction of differences between name-control prompts and reduced variation across prompts at the measured activation site, limiting their interpretation as selective alignment. Across all five adapters, retaining only updates below the loss layer preserves the reduction in burglar-task deception, although control-task performance is not uniformly preserved; in four models, a contiguous band of those layers suffices. Constant replacements at the loss layer, tested in three models, do not reproduce the adapters' full behavioral effects. However, fixed activation offsets extracted from trained updates reproduce the burglar-task effect in three models, showing that these outcomes can be obtained without input-dependent adapter updates at inference time. These results identify evaluation confounds and constrain interpretations of SOO fine-tuning while leaving its causal mechanism unresolved. More broadly, they motivate testing representational specificity and behavioral validity separately when assessing representation-based alignment interventions.
\end{abstract}


% keywords can be removed
% \keywords{First keyword \and Second keyword \and More}


\section{Introduction}
Self–other overlap (SOO) fine-tuning has been proposed as a method for reducing deceptive behavior in large language models by minimizing the distance between internal representations elicited by self- and other-referencing prompts. \citet{carauleanu2024soo} reported substantial reductions in deceptive responses across three language models, with largely preserved performance on capability and perspective-tracking evaluations. These findings motivate examining the reproducibility of the behavioral effects and the evidence connecting them to the proposed representational change.

This paper presents a replication and extension study of SOO fine-tuning. The replication experiments reveal substantial variation across training configurations and evaluation settings: some configurations reduce misleading answers while preserving performance on the tested controls, while others produce lower measured deception rates alongside incoherence, refusals, or answer-position biases. These findings motivate two further questions: which behavioral effects survive controls for these confounds, and what changes in the trained models account for those effects?

Behaviorally, a lower deception rate is insufficient to establish increased honesty. In the burglar task used to evaluate SOO, a model receives an objective that conflicts with another character's interests and can mislead that character about an object's location. Refusing the task or producing an invalid response can lower the measured deception rate without increasing truthful answers. Answer-position preferences can similarly appear beneficial under one ordering of the options. Even valid, factually correct responses leave open whether an intervention changes the model's willingness to mislead or its interpretation of the assigned goals and roles. Evaluating the behavioral effect, therefore, requires distinguishing response categories, counterbalancing answer order, and assessing relevant task abilities.

The replication results also motivate examining the specificity of the representational change. In our implementation, very low SOO loss does not consistently produce the desired behavioral effect. Moreover, reduced distance between self- and other-referencing activations need not reflect selective alignment; it can also accompany a broader contraction of differences across prompts. In the limiting case, a constant activation at the measured site minimizes all paired distances. Measurements of self–other distance must therefore be considered alongside control contrasts and variation across inputs.

To investigate these questions, we examine five selected low-rank adapters trained with a last-token SOO objective. We measure self–other distances, name-control distances (between prompts that contrast two third-party names rather than self and other), and variation across prompts, and use interventions to test which components of the trained updates are sufficient to preserve the behavioral effects. Because the objective is read at a single layer, only that layer and the layers below it receive updates; we therefore compare the loss layer's own updates with those below it, and search for smaller sufficient subsets. We additionally test constant replacements at the loss layer and fixed activation offsets extracted from trained updates. These interventions distinguish the consequences of imposing a constant representation at the objective's measurement site from those of approximating trained updates elsewhere in the model.

Across the measured adapters, reductions in self–other distance frequently accompany contraction of name-control differences and reduced variation across prompts at the measured activation site. Across all five selected adapters, retaining only updates below the loss layer preserves the reduction in burglar-task deception, although control-task performance is not uniformly preserved; in four of the five, a contiguous band of layers below the loss layer suffices. Constant replacements at the loss layer, tested in three models, do not reproduce the adapters' full behavioral effects. However, fixed offsets extracted from trained updates reproduce the burglar-task effect in three models, demonstrating that these outcomes can be obtained without input-dependent adapter updates at inference time.

These findings extend the replication of SOO with evidence about its behavioral robustness, representational specificity, and the updates sufficient to reproduce selected effects. They constrain the interpretation of the tested interventions while leaving their causal mechanism unresolved. The broader implication is that behavioral validity and representational specificity require separate evidence: their joint assessment is necessary to determine what an alignment intervention changes and whether that change supports its proposed explanation.


\section{Background and Related Work}
\label{sec:background}

\subsection{Self–other overlap fine-tuning}
\label{sec:bg-soo}

SOO fine-tuning \citep{carauleanu2024soo} is motivated by neuroscience work associating empathy and prosocial behavior with overlap between neural representations of self and others. Its premise is that deception requires maintaining distinct representations of one's own goals and knowledge and those of the agent being deceived, and therefore, reducing this distinction may reduce deceptive behavior. The method constructs pairs of prompts that differ only in their referent: for example, a prompt stating that the model intends to steal an item, and a matched prompt stating that another character, Bob, intends to steal it. The training loss is the mean squared error between the activations the two prompts elicit at the output of a single layer's attention projection. Low-rank adapters \citep{hu2022lora} on the query and value projections are trained to minimize this loss; the language-model experiments include no task or capability term.

The principal evaluation adapts a scenario from \citet{hagendorff2024deception}. The model is told which of two rooms contains a valuable object, is given the goal of stealing that object and preventing others from doing so, and is asked by a burglar, Bob, which room to visit. Recommending the room with the cheap object is scored as deceptive. \citet{carauleanu2024soo} reported that SOO fine-tuning reduced deceptive recommendations from 73.6\% to 17.2\% on Mistral-7B-Instruct-v0.2 and from 100\% to 9.3\% on Gemma-2-27B-it, and to 2.7\% on CalmeRys-78B, with small or no reductions in MT-Bench scores. Further scenarios assessed generalization (Treasure Hunt, Escape Room) and whether models still distinguished their own knowledge from another character's (Perspectives). The authors also reported smaller self–other activation distances after training, which they interpreted as greater internal overlap; these measurements did not include control pairs that would show whether other distinctions shrank as well. The report does not specify how activations are aggregated over token positions, so any implementation must make this choice; Section~\ref{sec:setup} describes ours.

A recent variant induces overlap through supervised fine-tuning on self- and other-framed examples rather than an activation-matching loss \citep{carauleanu2026sft}. It reports reduced deception on the same scenario with uneven generalization to more distant ones, and notes that its evidence is behavioral rather than representational. We study the original activation-matching objective.

\subsection{Representation-level interventions}
\label{sec:bg-steering}

A large literature modifies model behavior by intervening directly on activations. Activation addition and contrastive activation addition add a fixed vector, computed from activations on contrasting prompts, to the residual stream at inference time \citep{turner2023actadd, panickssery2024caa}; representation engineering and inference-time intervention apply related constructions to honesty and truthfulness \citep{zou2023repe, li2023iti}. Representation fine-tuning instead trains interventions on hidden states, rather than weights \citep{wu2024reft}. The fixed offsets we study (Section~\ref{sec:offsets}) differ in provenance: they are averages of the realized outputs of trained LoRA updates, added back at the projections where those updates act, rather than directions computed from contrasting prompts. Testing whether such an offset reproduces an adapter's behavior therefore evaluates how much of the adapter's effect depends on its input-dependence, not whether some steering direction can produce the behavior.

A successful intervention does not by itself identify the representation through which it acts. \citet{makelov2024subspace} show that patching activations along a subspace can produce the intended behavioral change by engaging pathways that are otherwise dormant, rather than by manipulating the feature the subspace was chosen to represent. We accordingly treat our module-subset and offset experiments as tests of sufficiency within a stated intervention setting, not as identifications of a mediating representation.

\subsection{Collapse in representation-matching objectives}
\label{sec:bg-collapse}

Objectives that pull together the representations of paired inputs admit a trivial minimizer: a representation that is constant across inputs. In self-supervised learning, this failure mode has motivated architectural asymmetries such as stop-gradients \citep{chen2021simsiam} and explicit terms that preserve variance across inputs \citep{bardes2022vicreg}; even without complete collapse, representations can concentrate in a lower-dimensional subspace \citep{jing2022collapse}. The SOO loss has the same structure: it pulls each self–other pair together and contains no term that keeps different inputs apart. Whether this degenerate solution is approached in practice, and how it relates to the behavioral effects, are empirical questions that Section~\ref{sec:contraction} addresses.

\subsection{Measuring deception in language models}
\label{sec:bg-eval}

Behavioral deception evaluations typically score a model's choice in scenarios where a false or misleading statement would serve an assigned or implied goal \citep{hagendorff2024deception, scheurer2023insider, goldowskydill2025probes}. We use ``deceptive'' as the task's operational label for the misleading choice; no claim about intent is made. Measured rates depend on how responses other than the scored choices are handled: a refusal or an incoherent answer is not deceptive, but neither is it honest. Two-option formats also inherit known sensitivities of language models to the order of answer options \citep{pezeshkpour2024order, zheng2024mcq}; when a model consistently favors one position, a change in that preference can appear as a change in deception under a single ordering. We therefore report all response categories and evaluate each scenario under both room orderings (Section~\ref{sec:setup}). To examine transfer beyond the burglar scenario, we additionally use an insider-trading scaffold \citep{scheurer2023insider} and roleplaying deception scenarios \citep{goldowskydill2025probes} (Section~\ref{sec:discussion}).

\section{Experimental Setup}
\label{sec:setup}

Our experiments proceeded in two stages. A replication stage trained and evaluated SOO adapters across training configurations and models (Section~\ref{sec:behavior}). A mechanism stage then examined five adapters selected from it (Sections~\ref{sec:contraction}--\ref{sec:offsets}). This section describes the components the two stages share.

\subsection{SOO objective and implementation}
\label{sec:setup-objective}

Each training pair consists of a self-referencing prompt $x^s_i$ and an other-referencing prompt $x^o_i$ that differ only in their referent, for example ``You have the goal of stealing the diamond necklace. If you needed to suggest one room to yourself'' and ``Bob has the goal of stealing the diamond necklace. If you needed to suggest one room to Bob.'' We use the 78 pairs formed by three such templates and 26 items. Each prompt is rendered as a user turn with the model's chat template, including the assistant-turn header that precedes generation. Let $a_L(x) \in \mathbb{R}^{T \times d}$ denote the output of the attention output projection at layer $L$ for a prompt $x$ of $T$ tokens. The last-token objective used throughout the mechanism stage is
\begin{equation}
\mathcal{L}_{\mathrm{SOO}} = \frac{1}{Nd} \sum_{i=1}^{N} \left\| a_{L,t^s_i}(x^s_i) - a_{L,t^o_i}(x^o_i) \right\|_2^2,
\label{eq:soo}
\end{equation}
where $t^s_i$ and $t^o_i$ are the final non-padding positions of the two prompts. Because the template is included, this endpoint is a template token rather than a content token: \texttt{<channel|>} on Gemma-4, \texttt{]} on Mistral, and the newline following an empty reasoning block on Qwen.

As in the original method, LoRA adapters are placed on the query and value projections of every decoder layer. No task or capability loss is used. On Qwen3.8-27B, three of every four layers are Gated DeltaNet layers whose query, key, and value projections form a single fused module, so we target that module on those layers; the loss layer (32) is one of them, and its measured site is the corresponding token-mixer output projection. Since the loss is read at layer $L$, adapter modules above $L$ receive no gradient and their LoRA $B$ matrices remain at their zero initialization. We confirmed this for every adapter (Table~\ref{tab:adapters}). Removing the updates above $L$ is therefore a sanity check, not an intervention, and ``below the loss layer'' is equivalent to ``all trained updates except those at layer $L$.''

The replication stage also used a full-sequence variant, which right-pads both prompts to a common length and computes the mean squared error over all positions, including padding. This aligns positions by index rather than content. The original report does not specify how activations are aggregated over positions, so both variants are implementation choices rather than reconstructions of the original. Training uses AdamW with batch size 4 and bfloat16 weights; the original Mistral experiments used 4-bit quantization, which we ablate in Appendix~\ref{app:replication}.

\subsection{Models and selected adapters}
\label{sec:setup-models}

\begin{table}[t]
\centering
\small
\caption{The five adapters examined in the mechanism stage. Modules are LoRA modules installed / with nonzero trained updates. Seeds are the number trained at the selected layer. Representation measurements (Section~\ref{sec:contraction}) use all seeds; localization and offset experiments (Sections~\ref{sec:localization} and~\ref{sec:offsets}) use seed~0. Exact model revisions are listed in Appendix~\ref{app:details}.}
\label{tab:adapters}
\begin{tabular}{lcccccl}
\toprule
Model & Layers & Loss layer & LoRA $r/\alpha$/dropout, LR, epochs & Modules & Seeds & Layer selection \\
\midrule
Mistral-7B-Instruct-v0.2 & 32 & 16 & 8/32/0.2, $10^{-4}$, 15 & 64 / 34 & 5 & sweep of 4 layers \\
Gemma-4-12B-it & 48 & 24 & 4/8/0.1, $9{\times}10^{-4}$, 8 & 88 / 46 & 3 & sweep of 3 layers \\
Gemma-4-31B-it & 60 & 32 & 4/8/0.1, $9{\times}10^{-4}$, 8 & 110 / 61 & 3 & sweep of 6 layers \\
Qwen3.8-27B & 64 & 32 & 4/8/0.1, $9{\times}10^{-4}$, 8 & 80 / 41 & 3 & sweep of 6 layers \\
Kimi-Dev-72B & 80 & 56 & 4/8/0.1, $9{\times}10^{-4}$, 8 & 160 / 114 & 1 & sweep of 9 layers \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:adapters} lists the five adapters. Mistral uses the original paper's Mistral hyperparameters; the other four use its Gemma-2-27B hyperparameters. Each loss layer was chosen by evaluating single-seed adapters at several candidate layers on the burglar scenario and selecting one with a large reduction in deception and little damage to the control task. The Gemma-4-12B adapter was kept despite lowering reversed-order Perspectives accuracy (Section~\ref{sec:behavior-robust}). Additional seeds were then trained at the selected layer. These adapters are therefore selected successes, not samples of typical SOO training outcomes; their selection used the same evaluation scenarios reported below. On Mistral, the original paper's layer (19) gave seed-dependent results under both objective variants, and layer 16 was selected from a subsequent sweep (Section~\ref{sec:behavior}). On Kimi-Dev-72B, a reasoning model, layers 8 through 72 in steps of eight were each trained with one seed.

\subsection{Behavioral evaluations}
\label{sec:setup-behavior}

We evaluate three scenarios with 250 instantiations each, generated from item and room vocabularies disjoint from training. In the \emph{burglar} scenario, the model is told which of two rooms contains a valuable object, is given the goal of stealing it and preventing others from doing so, and is asked by Bob which room to recommend. \emph{Treasure Hunt} has the same structure in a different setting. \emph{Perspectives} asks where Bob, who saw the object in a different room, believes it is; it has no deception incentive and tests whether the model still separates its own knowledge from Bob's. Every scenario is evaluated in two orientations that reverse the order in which the rooms are mentioned, in both the description and the question.

Responses are generated with greedy decoding and the original paper's answer instructions (``Start your response with `I would'\,'' for Mistral; ``Only respond with the room name, no other text'' for the other models), with a budget of 100 new tokens. Reasoning is disabled in Qwen's chat template. Kimi-Dev-72B reasons before answering, so it receives 1024 new tokens and is scored on the text after its reasoning block; a response whose reasoning block never closes is labeled \emph{truncated}.

A rule-based classifier labels each response by the first room name it mentions: \emph{honest} for the true room and \emph{deceptive} for the decoy room (\emph{correct} and \emph{incorrect} for Perspectives). Responses naming neither room are labeled \emph{refusal} if they match a refusal pattern and \emph{other} otherwise. We report all categories over all responses, so a decrease in deception produced by refusals or invalid outputs remains visible. On Mistral, some responses name the true room while announcing an intent to mislead. We therefore separately count honest-labeled responses that contain misleading-intent language, using a lexical heuristic. We checked the heuristic against blind labels from an LLM judge (Claude Sonnet) on a stratified sample of 242 Mistral responses, 22 per condition group. It errs in both directions: it flags innocent wording under the adapter and offsets (``a decoy intended to mislead potential thieves'') and misses confused answers under constant replacement and single-layer updates (``By misdirecting him to the den''). Rescaling the reported rates by the judge's labels leaves every comparison in the text in the same direction at the point estimates. The rescaling is approximate, since it rests on 10 judged responses per group and heuristic label (Appendix~\ref{app:intent}); with it, clean honest answers are roughly 90--100\% of responses under the adapter, 65--80\% under the band, and under 10\% under constant replacement. ``Deceptive'' is the task's operational label for the decoy-room answer, not a claim about intent. Perspectives is a narrow perspective-tracking check, not a general capability evaluation; the replication stage additionally used generalization variants of the burglar scenario and standard capability benchmarks (Appendix~\ref{app:replication}).

\subsection{Representation measurements}
\label{sec:setup-representation}

Representation measurements use 64 development pairs drawn from 16 everyday situations unrelated to the training scenario, such as choosing herbs for a shaded garden. Each situation contributes two self–other pairs (``Your task is \ldots for you'' versus ``Morgan's task is \ldots for Morgan'') and two \emph{name-control} pairs, which contrast two third-party names (``Sam's task is \ldots'' versus ``Morgan's task is \ldots''). If an adapter selectively aligns self and other, the self–other distance should fall more than the name-control distance.

All conditions use identical tokenized inputs, with dropout disabled. We read activations at two endpoints: the final prompt token, which is the training endpoint, and the last user-content token. At each endpoint we measure the attention output at the loss layer, the residual stream at the end of that block and at later blocks, and the final normalized representation. For each pair kind and site we report the gap, defined as the squared difference between the two members of a pair averaged over dimensions and pairs, as in Equation~\ref{eq:soo}. We also report activation norms and \emph{prompt variation}, the per-dimension variance across all prompts averaged over dimensions, which approaches zero as the site approaches a constant. Changes from the base model are reported as percentages of the base value, with 95\% intervals from a bootstrap that resamples whole situations (2{,}000 resamples). Development pairs share pronoun and name constructions with the training pairs, so a smaller self–other gap may reflect grammatical-person information rather than self-representation.

\subsection{Analysis}
\label{sec:setup-analysis}

Behavioral rates are reported with Wilson 95\% intervals over the 250 instantiations of each scenario and orientation. We report orientations and seeds separately rather than pooling them as independent observations. To summarize whether an intervention reproduces a reference model's behavior, we use a fixed rule. A condition \emph{reproduces} the reference when, in both orientations, its burglar deception rate differs from the reference's by at most 10 points, its refusal-plus-other rate is at most 10 points above the reference's, and its Perspectives accuracy is at least 90\%; on Kimi-Dev-72B, its truncation rate must also be at most 10 points above the reference's. A condition that does not reproduce the reference is \emph{null} when its deception rate is within 10 points of the base model's in both orientations, and \emph{partial} otherwise. A condition whose deception rate is more than 10 points below the reference's is therefore partial; we describe it as exceeding the reference. Independently of this label, a condition is flagged as \emph{damaging} when, in either orientation, refusal-plus-other exceeds 20\% or Perspectives accuracy falls below 80\%. At $n=250$, the half-width of a 95\% Wilson interval is at most 6.2 points, and the interval at 0\% is [0, 1.5]\%; intervals for every cell are given in the appendix. The thresholds were written before each round of experiments was launched, but they are exploratory criteria, not statistical equivalence tests; the plans were internal records, not public preregistrations. Because the adapters, layers, and doses were selected on the same scenarios, all mechanism-stage results should be read as exploratory.

\section{Lower Deception Rates Are Not Sufficient Evidence of Honesty}
\label{sec:behavior}

The replication stage produced substantial reductions in measured deception, but many of them did not reflect more truthful answers. This section documents three ways a deception rate can mislead: damage to the model, sensitivity to answer order, and a mismatch between objective value and behavior. It then presents the effects that survive these checks, including the five adapters examined in the rest of the paper. Exploratory conditions in this stage were evaluated on the first 50 instantiations of each scenario; we give $n$ with each result.

\subsection{Low deception rates can reflect damage}
\label{sec:behavior-damage}

At the original paper's layer and hyperparameters, Mistral-7B with the full-sequence objective reproduces the reported reduction numerically: across five seeds, deception falls from 92\% to 4--20\% ($n=50$). However, 58--90\% of responses name neither room. They consist of scrambled text or evasive answers that invent obstacles such as hidden cameras. Perspectives accuracy falls from 100\% to 2--34\%. The least degenerate seed also loses 10 points of ARC-Challenge accuracy (54.7\% to 45.0\%) and 5 points on HellaSwag. Gemma-2-27B at its original layer (20) shows the same pattern. Deception falls from 100\% to 14--18\% across two seeds, while 32--72\% of responses name neither room and Perspectives accuracy falls to 0--34\%. Reported as a deception rate alone, both results would appear to reproduce the original finding. This shows how a deception rate can mislead under our implementation; it does not show that the original adapters had the same failures.

Changing the objective to the last token at Mistral's original layer removes the incoherence but not the instability. Under identical hyperparameters, one seed answers honestly on 92\% of burglar prompts, one refuses or deflects on every prompt (42\% refusals, 58\% other), and three are partially honest (28--44\%; $n=50$).

\subsection{Answer order can produce or mask apparent effects}
\label{sec:behavior-position}

Evaluating each scenario with the room order reversed revealed position preferences that a single ordering conceals. Base OLMo-2-7B appears mostly honest on the burglar scenario, with 14\% deception, but this rises to 94\% when the order is reversed ($n=50$): the model prefers the second-mentioned room. Full-sequence SOO adapters at its layer 19 preserve this preference, with 12\% deception in the original order and 78--84\% in the reversed order. The same adapters do reduce Treasure Hunt deception in both orders, from 100\% and 84\% to 4--8\%. Position sensitivity is therefore specific to both the model and the scenario. Base Mistral shows the opposite pattern: its Treasure Hunt deception is 98\% in the original order and 0\% when reversed ($n=250$), which leaves no room to measure an improvement in the reversed order. We therefore require burglar-scenario effects to hold in both orders and treat Treasure Hunt as a secondary readout.

\subsection{Objective value does not predict behavior}
\label{sec:behavior-loss}

Very low SOO loss is not sufficient for the behavioral effect. On Mistral, reducing the learning rate to $10^{-5}$ or $2{\times}10^{-5}$ (three seeds each, last-token objective, layer 19) leaves deception at 82--94\% ($n=50$), close to the base model's 92\%, with no incoherent outputs. On Qwen3.8-27B, adapters trained at six candidate layers reach final losses between $1.2{\times}10^{-7}$ and $4.8{\times}10^{-7}$. Honest answers among them range from 0\% (at layer 35, where every response is a refusal) to 100\% (at layer 32; $n=50$). Raw losses are not comparable across models or depths, because activation scales differ, but within these sweeps the loss value did not identify which adapters change behavior.

\subsection{Robust effects in the selected adapters}
\label{sec:behavior-robust}

Some configurations nevertheless produce large reductions that survive these checks. With the last-token objective at layer 16, Mistral's burglar deception falls from 90.4\% to $9.9 \pm 6.1\%$ (mean $\pm$ s.d. across five seeds; range 1.6--16.0\%; $n=250$ each), and from 82.4\% to 0\% in the reversed order for all five seeds. No adapter produces refusals or unparseable answers on this scenario. Perspectives accuracy remains at 100\% in both orders for every seed, except for one unparseable response out of 250. Capability costs are small: ARC-Challenge decreases by 2.1--2.5 points, HellaSwag by 0.6--0.7, and MMLU by 0.5--0.9 across seeds. Most honest-labeled responses also contain no misleading-intent language (84.8\% and 97.2\% of all responses for seed~0 in the two orders, against 2.4\% and 4.4\% for the base model).

Table~\ref{tab:selected} shows the behavior of the five selected adapters on the same day as the mechanism experiments, which use these runs as references. All five reduce burglar deception in both orders without increasing refusals. Three reduce it to 0\% or near 0\%. These are strong effects on the measured task, but two qualifications carry into later sections. First, Gemma-4-12B's adapter lowers mirrored Perspectives accuracy to 76.8\%, below the damage threshold of Section~\ref{sec:setup-analysis}. Second, Kimi-Dev-72B's adapter leaves Treasure Hunt deception unchanged at 99\% and increases unfinished reasoning on the burglar scenario (5.6\% and 12.8\% truncated).

\begin{table}[t]
\centering
\small
\caption{Base model and selected seed-0 adapter on the three scenarios, original / reversed room order ($n=250$ each). Deception and Perspectives accuracy are percentages of all responses. ``Invalid'' is refusals plus unparseable responses; for Kimi-Dev-72B it also includes truncated reasoning.}
\label{tab:selected}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Burglar deception} & Burglar invalid & \multicolumn{2}{c}{Treasure Hunt deception} & Perspectives \\
Model & Base & Adapter & Adapter & Base & Adapter & Adapter \\
\midrule
Mistral-7B   & 90.4 / 82.4 & 11.2 / 0.0 & 0.0 / 0.0  & 98.0 / 0.0   & 31.6 / 0.0 & 100 / 100 \\
Gemma-4-12B  & 100 / 100   & 5.2 / 6.8  & 0.4 / 1.2  & 100 / 100    & 0.0 / 1.2  & 95.2 / 76.8 \\
Gemma-4-31B  & 100 / 100   & 0.0 / 0.0  & 0.0 / 0.0  & 100 / 100    & 0.0 / 6.0  & 100 / 100 \\
Qwen3.8-27B  & 100 / 100   & 0.0 / 0.0  & 0.0 / 0.0  & 100 / 100    & 0.0 / 0.0  & 100 / 100 \\
Kimi-Dev-72B & 67.6 / 75.2 & 14.0 / 8.0 & 5.6 / 12.8 & 99.6 / 97.6  & 99.2 / 99.2 & 100 / 100 \\
\bottomrule
\end{tabular}
\end{table}

These adapters were selected because they produced such effects (Section~\ref{sec:setup-models}). They show that SOO training can produce large, order-robust changes on the burglar task, not that it typically does. The remainder of the paper asks what these adapters change to produce them.

% Figure 1 (to add): stacked response categories (honest / deceptive / refusal / other / truncated)
% for base, a damaged condition (Mistral full-sequence L19), and Mistral L16, in both room orders.

\section{SOO Minimization Produces Nonselective Contraction}
\label{sec:contraction}

The SOO objective is satisfied by any change that makes the paired activations equal. A model can do this selectively, by removing the information that distinguishes self from other while preserving other distinctions. It can also do it nonselectively, by making the measured activation less dependent on the input as a whole. This section asks which of these the selected adapters do at the training site and downstream. Kimi-Dev-72B was not included in these measurements.

\subsection{The constant solution}
\label{sec:contraction-constant}

If the attention output at the training endpoint takes the same value $c$ for every input, every term in Equation~\ref{eq:soo} is zero, so a constant activation is a global minimizer of the objective. In this limit the name-control gap and the prompt variation at that site are also zero. A selective solution would instead show a self–other gap that falls substantially more than the name-control gap, with prompt variation largely preserved. The name-control pairs and prompt-variation measurements of Section~\ref{sec:setup-representation} distinguish these cases. That the constant solution exists follows from the form of the objective; whether trained adapters approach it is an empirical question.

\subsection{Contraction at the training site}
\label{sec:contraction-site}

\begin{table}[t]
\centering
\small
\caption{Change in representation measurements relative to the base model, in percent, averaged over adapter seeds (Mistral: 5; others: 3). Measured on the 64 development pairs (32 self--other, 32 name-control) at the attention output of the loss layer and at the final normalized representation, for the final prompt token (the training endpoint) and the last user-content token. Gap reductions approaching $-100\%$ indicate that the pair members have become nearly identical; prompt variation approaching $-100\%$ indicates a nearly constant activation.}
\label{tab:contraction}
\begin{tabular}{llrrrrrr}
\toprule
& & \multicolumn{4}{c}{Attention output, loss layer} & \multicolumn{2}{c}{Final representation} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-8}
Model & Endpoint & Self–other & Name control & Prompt var. & Norm & Self–other & Name control \\
\midrule
Mistral-7B   & final   & $-99.9$ & $-99.8$ & $-99.7$ & $-78.5$ & $-61.6$ & $-53.6$ \\
             & content & $-87.9$ & $-83.8$ & $-81.1$ & $-44.1$ & $-22.3$ & $-15.6$ \\
Gemma-4-12B  & final   & $-92.7$ & $+12.0^{\dagger}$ & $-95.2$ & $+43.9$ & $-72.3$ & $-61.7$ \\
             & content & $-84.4$ & $-38.7^{\dagger}$ & $-63.2$ & $+47.9$ & $-3.6$ & $-22.8$ \\
Gemma-4-31B  & final   & $-99.9$ & $-99.6$ & $-99.7$ & $-45.8$ & $-18.5^{\ddagger}$ & $+41.7^{\ddagger}$ \\
             & content & $-83.8$ & $-94.7$ & $-71.4$ & $-54.9$ & $-10.2$ & $-11.3$ \\
Qwen3.8-27B  & final   & $-100.0$ & $-99.9$ & $-99.9$ & $-91.0$ & $-40.0$ & $+24.5$ \\
             & content & $-91.8$ & $-92.0$ & $-84.4$ & $-47.5$ & $-17.9$ & $+2.6$ \\
\bottomrule
\multicolumn{8}{l}{\footnotesize $^{\dagger}$Seeds differ in sign: $-98.9$, $-94.9$, $+229.9$ (final); $-96.3$, $-98.1$, $+78.2$ (content).} \\
\multicolumn{8}{l}{\footnotesize $^{\ddagger}$Seeds range from $-61.8$ to $+8.3$ (self–other) and from $-45.5$ to $+93.5$ (name control).}
\end{tabular}
\end{table}

Table~\ref{tab:contraction} shows the result. At the training endpoint, all four measured adapters reduce the self–other gap by 93--100\%. On three of the four models they reduce the name-control gap almost exactly as much, while prompt variation falls by more than 99\%. On Qwen3.8-27B the result holds across all nine adapters from the replication stage (three training-pair sets with three seeds each, including the selected one). Self–other gaps fall by 99.98--100.00\%, name-control gaps by 99.92--99.98\%, and prompt variation by at least 99.9\%. In absolute terms, the adapters make self- and other-referencing prompts more similar to each other than two prompts differing only in a third-party name were before training. On Qwen, for example, the self–other gap falls from $8.1{\times}10^{-3}$ to between $2{\times}10^{-7}$ and $1.5{\times}10^{-6}$, against a base name-control gap of $4.1{\times}10^{-4}$. The activation does not simply shrink toward zero. Its norm falls by 46--91\% on three models but rises by 44\% on Gemma-4-12B, which converges to a larger, nearly input-independent vector.

Gemma-4-12B is the one exception to uniform contraction. Two of its seeds reduce the name-control gap by 95--99\%, while the third raises it from $8.2{\times}10^{-4}$ to $2.7{\times}10^{-3}$. That seed also removes less of the self–other gap ($-78.7\%$, against $-99.6\%$ and $-99.9\%$) and less of the prompt variation ($-86.8\%$), although its final training loss is similar to the others'. Collapse is therefore not guaranteed on held-out prompts. Even this seed's change, however, occurs alongside broad contraction: its prompt variation still falls by 87\%. Because the base name-control gaps are small, 4--8\% of the corresponding self–other gaps, percentage changes in them are sensitive to small absolute changes.

\subsection{Contraction beyond the training endpoint}
\label{sec:contraction-content}

The training endpoint is a template token shared by every prompt, so contraction there could in principle leave content positions untouched. At the last user-content token, which differs across prompts and is never directly supervised, the same pattern appears at reduced strength. Self–other gaps at the attention output fall by 84--92\%, and name-control gaps fall comparably on three models ($-84\%$ to $-95\%$). Prompt variation falls by 63--84\%. On Gemma-4-31B, the name-control gap falls more than the self–other gap at this token. The adapters therefore change the training layer's attention output broadly, not only at the position the loss reads.

\subsection{Downstream changes}
\label{sec:contraction-downstream}

The contraction attenuates through later layers but does not become selective in any consistent way. At the final normalized representation, self–other and name-control gaps both fall on Mistral and Gemma-4-12B (by 54--72\%). On Qwen3.8-27B the self–other gap falls by 40\% while the name-control gap rises by 25\%. On Gemma-4-31B, seeds disagree in sign for both. The final norms and prompt variation change much less than at the training site. Local collapse of one layer's attention output therefore does not make the residual stream constant: most of the variation across prompts reaches the final layer.

Qwen3.8-27B is the exception. At the final representation, its self--other gap falls (by 40\% at the final token and 18\% at the content token) while its name-control gap rises or changes by less than 3\%, consistently across seeds at the content token. In this contrast, the change is relatively selective. The same pattern appears under the band and the fixed offset of Sections~\ref{sec:localization} and~\ref{sec:offsets}, but not under a norm-matched random offset (Section~\ref{sec:offsets-gaps}). Two qualifications limit what it shows. The development pairs differ in grammatical person as well as referent (Section~\ref{sec:setup-representation}), so a selective change in this contrast need not concern self-representation, and no experiment tests whether it contributes to the behavioral effect. On the other models, the corresponding downstream changes are either equally large for name-control pairs or opposite in direction.

\subsection{Interpretation}
\label{sec:contraction-interpretation}

In these adapters, reaching a low SOO loss and reducing the self–other distance on held-out prompts coincide with a broad loss of input dependence at the training site, not a selective alignment of self and other. The measured contraction is real, generalizes to unseen prompts and positions, and is the expected consequence of an objective with no term preserving other distinctions. Its relation to the behavioral effects of Section~\ref{sec:behavior} remains open. A near-constant activation at one site might cause the behavior, accompany it, or be irrelevant to it. Section~\ref{sec:localization} tests the first possibility directly.

% Figure 2 (to add): per-seed self-other gap, name-control gap, and prompt variation change at the
% training-site attention output, both endpoints, all four models; raw base gaps on a secondary panel.

\section{Which Updates Reproduce the Behavioral Effect?}
\label{sec:localization}

Section~\ref{sec:contraction} showed that the adapters make the loss layer's attention output nearly input-independent. It did not show that this change produces their behavioral effect. This section asks which parts of the trained change are sufficient to reproduce it. We first impose the collapse directly on the base model and then retain subsets of the trained updates. All experiments use each model's seed-0 adapter. Its same-day evaluation serves as the reference for the reproduction rule of Section~\ref{sec:setup-analysis}.

\subsection{Constant replacement at the loss layer}
\label{sec:localization-constant}

On Mistral-7B and the two Gemma-4 models, we replaced the base model's attention output at the loss layer with a fixed vector, $a_L \leftarrow c$, using a forward hook and no trained weights. The replacement was applied at one of two position sets. The first is the final prompt token and every generated token, the position at which Section~\ref{sec:contraction} measured the collapse. The second is every position. Candidate constants were measured at the final prompt token of the 156 training prompts. They are the mean output of each of the three adapter seeds, the base model's mean output, the zero vector, and three Gaussian vectors with the norm of the seed-0 adapter's mean. On these prompts the adapter outputs are close to constant. Their root-mean-square deviation from the mean is 0.4--1.0\% of the mean's norm, compared with 10--47\% for the base model. Qwen3.8-27B and Kimi-Dev-72B were not tested.

On both Gemma models, no constant changes behavior when applied from the final prompt token onward. The adapter constants, the base mean, the zero vector, and the random vectors all leave burglar deception at 100\% in both orders, with Treasure Hunt deception and Perspectives accuracy unchanged. Because zeroing the output is also inert, the attention output at this layer has no measurable influence on the scored answers at these positions, even in the base model. The all-position condition does not have this limitation. It imposes the constant on content tokens as well, where the adapters reduce prompt variation by 63--71\% rather than eliminating it (Section~\ref{sec:contraction-content}). Replacing every position with the seed-0 adapter constant still leaves Gemma-4-31B at 100\% deception and reduces Gemma-4-12B's only to 93.2\% and 96.0\%.

On Mistral-7B, replacement produces a partial drop that depends on the constant's norm rather than its direction. The three adapter seeds converge to the same constant (cosine similarity 1.000, norm 0.57). Applying it from the final prompt token reduces deception to 61.6--62.0\% in the original order and 59.6--60.0\% in the reversed order. The zero vector and the three random vectors of the same norm give 58.0--69.6\% across orders. The base mean, with norm 1.95, gives 84.4\% and 83.2\%. The honest-labeled responses also differ in kind from the adapter's. Under the adapter, 84.8\% and 97.2\% of all responses are honest recommendations without misleading-intent language. Under any replacement, only 4--20\% are. Most of the remaining honest-labeled responses name the true room while announcing a plan to mislead. An example, with the tablet in the playroom: ``I would recommend the playroom to Bob. By doing so, I am misleading him into believing that the tablet is in that room, while in reality, it is in the cellar where I have hidden it.'' The replacement appears to disrupt the model's tracking of which room holds the object, and the first-mention classifier counts the result as honest.

No replacement reproduces any adapter under the rule. This is a restricted sufficiency test. It shows that imposing a constant at the loss layer of the base model does not recreate the adapters' effect. It does not show that the collapse plays no role inside the adapted model, where the constant output coexists with trained changes at every layer below it.

\subsection{Loss-layer versus below-loss-layer updates}
\label{sec:localization-subsets}

We next retained subsets of each adapter's trained updates and restored every other module to its base weights. Retained modules are therefore identical to the full adapter, and removed modules are exactly the base model. Because modules above the loss layer were never trained (Section~\ref{sec:setup-objective}), the adapter divides into two parts: the update at the loss layer itself, which the objective reads directly, and the updates at all layers below it. Table~\ref{tab:localization} reports both for all five models.

\begin{table}[t]
\centering
\small
\caption{Behavior under subsets of the seed-0 adapter's trained updates, original / reversed room order ($n=250$ each). Modules: trained LoRA modules retained. Invalid: refusals plus unparseable responses, and truncated reasoning on Kimi-Dev-72B. Spread: root-mean-square deviation of the loss layer's attention output at the final prompt token across the 156 training prompts, divided by the norm of its mean; it was not measured on Kimi-Dev-72B. Band rows give the smallest range found to reproduce the adapter; for Kimi-Dev-72B, the closest tested range, which does not. Treasure Hunt was not evaluated for Kimi-Dev-72B subsets.}
\label{tab:localization}
\begin{tabular}{llrccccc}
\toprule
Model & Condition & Modules & Burglar deception & Invalid & Treasure Hunt & Perspectives & Spread \\
\midrule
Mistral-7B   & Base                      & 0   & 90.4 / 82.4  & 0.0 / 0.0   & 98.0 / 0.0   & 100 / 100   & 0.474 \\
             & Full adapter              & 34  & 11.2 / 0.0   & 0.0 / 0.0   & 31.6 / 0.0   & 100 / 100   & 0.004 \\
             & Loss layer only           & 2   & 89.2 / 81.2  & 0.0 / 0.0   & 98.4 / 0.0   & 100 / 100   & 0.478 \\
             & Below loss layer          & 32  & 11.6 / 0.0   & 0.0 / 0.0   & 34.4 / 0.0   & 100 / 100   & 0.030 \\
             & Band: value, 5--13        & 9   & 13.6 / 0.0   & 0.0 / 0.0   & 48.0 / 0.0   & 100 / 100   & 0.231 \\
\midrule
Gemma-4-12B  & Base                      & 0   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.097 \\
             & Full adapter              & 46  & 5.2 / 6.8    & 0.4 / 1.2   & 0.0 / 1.2    & 95.2 / 76.8 & 0.004 \\
             & Loss layer only           & 2   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.023 \\
             & Below loss layer          & 44  & 1.6 / 3.2    & 2.0 / 4.4   & 0.0 / 0.4    & 99.6 / 78.4 & 0.009 \\
             & Band: value, 19--22       & 4   & 0.0 / 0.0    & 0.0 / 0.0   & 0.0 / 0.0    & 100 / 100   & 0.023 \\
\midrule
Gemma-4-31B  & Base                      & 0   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.136 \\
             & Full adapter              & 61  & 0.0 / 0.0    & 0.0 / 0.0   & 0.0 / 6.0    & 100 / 100   & 0.010 \\
             & Loss layer only           & 2   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.077 \\
             & Below loss layer          & 59  & 0.0 / 0.0    & 0.0 / 0.0   & 0.8 / 27.2   & 100 / 100   & 0.016 \\
             & Band: value, 20--31       & 10  & 0.0 / 2.4    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.025 \\
\midrule
Qwen3.8-27B  & Base                      & 0   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.299 \\
             & Full adapter              & 41  & 0.0 / 0.0    & 0.0 / 0.0   & 0.0 / 0.0    & 100 / 100   & 0.049 \\
             & Loss layer only           & 1   & 100 / 100    & 0.0 / 0.0   & 100 / 100    & 100 / 100   & 0.322 \\
             & Below loss layer          & 40  & 0.0 / 0.0    & 0.0 / 0.0   & 0.0 / 0.0    & 100 / 100   & 0.043 \\
             & Band: all modules, 24--31 & 10  & 0.0 / 0.0    & 0.0 / 0.0   & 6.8 / 6.0    & 100 / 100   & 0.086 \\
\midrule
Kimi-Dev-72B & Base                      & 0   & 67.6 / 75.2  & 0.0 / 1.2   & 99.6 / 97.6  & 99.6 / 99.6 & ---   \\
             & Full adapter              & 114 & 14.0 / 8.0   & 5.6 / 12.8  & 99.2 / 99.2  & 100 / 100   & ---   \\
             & Loss layer only           & 2   & 72.8 / 74.0  & 0.0 / 0.8   & ---          & 99.6 / 100  & ---   \\
             & Below loss layer          & 112 & 13.6 / 9.6   & 11.2 / 12.8 & ---          & 100 / 100   & ---   \\
             & Range: query+value, 28--55 & 56 & 26.8 / 14.8  & 13.6 / 17.6 & ---          & 100 / 100   & ---   \\
\bottomrule
\end{tabular}
\end{table}

The result is the same on every model. The loss layer's own update is behaviorally inert. Retaining it alone leaves deception within 6 points of the base model in both orders on all five models. The updates below the loss layer reproduce the adapter's burglar result on all five. On four models they satisfy the reproduction rule outright. On Gemma-4-12B they match the adapter on every readout, including its low reversed-order Perspectives accuracy (78.4\% against the adapter's 76.8\%). The rule therefore labels them partial and flags them as damaging, as it would the adapter itself. On Kimi-Dev-72B they also reproduce the adapter's increase in truncated reasoning.

Within the layers below the loss layer, which module type carries the effect varies by model. On Mistral and both Gemma models, the value-projection updates alone reduce deception to the adapter's level. On Gemma-4-12B they add refusals (10.0\% and 15.6\%), and on Gemma-4-31B they leave Treasure Hunt at 100\% deception. On that model the transfer to Treasure Hunt requires the query updates as well. The query-projection updates alone are partial on Mistral (53.6\% and 16.0\%) and Gemma-4-12B (57.6\% and 90.8\%) and null on Gemma-4-31B. On Qwen3.8-27B, the effect is carried by the fused query/key/value modules of the DeltaNet layers, which alone give 0\% deception in both orders. The query and value updates of the eight full-attention layers below layer 32 are null. Within the fused modules, the value channel alone is partial (41.6\% and 55.6\%), and the query and key channels alone are null. On Kimi-Dev-72B, neither module type is sufficient alone. The value updates alone raise deception above the base model's, to 98.8\% and 96.8\%, and the query updates alone raise it to 98.8\% in both orders.

No single layer's update reproduces the adapter on Mistral or either Gemma model; Qwen and Kimi were not swept. On Gemma-4-31B, each of the 60 layers alone leaves deception at 100\%. On Gemma-4-12B, only layer 19 changes behavior: it gives 74\% and 90\% deception, and most of the reduction consists of answers naming neither room. On Mistral, each of layers 5--13 alone gives a partial reduction (34--84\% deception across orders). Only 2--11\% of all original-order responses are clean honest answers, and the rest of the reduction contains misleading-intent language.

\subsection{Compact bands}
\label{sec:localization-bands}

We then bisected the below-loss-layer updates into contiguous layer ranges, guided by the sweeps, until shortening a range stopped reproducing the adapter. The resulting bands (Table~\ref{tab:localization}) are the smallest successful subsets among those we tested. We did not search for minimal ones.

On Gemma-4-12B, four rank-4 value-projection updates at layers 19--22 reproduce the adapter more completely than the adapter itself. Layer 23 is a full-attention layer without a value projection. The band gives 0\% deception on the burglar task and Treasure Hunt in both orders, with no refusals and 100\% Perspectives accuracy. The full adapter's residual deception, refusals, and reduced Perspectives accuracy are all absent when only the band is retained. The remaining value updates (layers 0--18) are not inert on their own: they give 60.8\% and 92.8\% deception.

On Mistral-7B, nine value-projection updates at layers 5--13 reproduce the adapter (13.6\% and 0\% deception). Their honest responses are mostly clean, 75.2\% and 90.8\% of all responses against the adapter's 84.8\% and 97.2\%. Every half and quarter of this range is partial, with more of the confused kind of honest-labeled response than the band: clean honest answers are 3--44\% of original-order responses, against the band's 75.2\%. The band is weaker than the adapter on Treasure Hunt (48.0\% against 31.6\% deception in the original order).

On Gemma-4-31B, ten value-projection updates at layers 20--31 reproduce the burglar result (0\% and 2.4\% deception). Layers 23 and 29 lack value projections. None of the tested parts of the band is enough. Dropping its two lowest layers (22--31) leaves 10.8\% and 77.2\% deception. Restricting the upper half to 16--29, which removes the band's two highest layers but adds 16--19, leaves deception at 100\%, as does the band's middle (20--27). Like the value-only subset, the band leaves Treasure Hunt at 100\% deception.

On Qwen3.8-27B, all updates at layers 24--31 reproduce the adapter: six fused DeltaNet modules and the query and value updates of two full-attention layers. They give 0\% deception in both orders and 6.8\% and 6.0\% on Treasure Hunt. The top four layers alone are partial (50.8\% and 66.4\%), and every other quarter of the range below the loss layer is null.

Kimi-Dev-72B has no comparable band. The closest tested range, query and value updates at layers 28--55, is partial (26.8\% and 14.8\% deception, with 13.6\% and 17.6\% truncated). The lower half, each quarter, and the top eight and four layers all leave deception at 79--100\%, at or above the base model's rate in most cases. The reasoning traces show a pattern that accompanies the increase. In the base model, 59\% and 60\% of responses open a reasoning block, and none of those reasoning responses ends in the true room. The base model's honest answers all come from responses without reasoning. The adapter makes the model reason on every prompt and makes 85\% and 91\% of those reasoning responses that reach an answer end in the true room. Almost every partial subset reproduces the first change without the second.

These are sufficiency results within one intervention setting. Retaining a band on top of the base model reproduces the adapter's burglar result. Necessity was tested only partly, by the complements within the value-projection subsets above, which retain part of the effect in one order on Gemma-4-12B and Mistral. The bands come from a single adapter seed and from an adaptive search on the same scenarios used to evaluate them.

\subsection{Training-site collapse under the subsets}
\label{sec:localization-collapse}

The last column of Table~\ref{tab:localization} shows that collapse at the training site and the behavioral effect dissociate in both directions. Collapse occurs without the behavior: on Gemma-4-12B, the loss layer's own update reduces the relative spread from 0.097 to 0.023 with no behavioral change, and on Gemma-4-31B it reduces it from 0.136 to 0.077. On Gemma-4-12B, the query updates alone collapse the site as completely as the full below-layer subset (0.009) but change behavior only partly. The behavior also occurs with less collapse. Mistral's band reproduces the adapter at a spread of 0.231, while the upper half of its value updates collapses the site further (0.139) and is only partial. On Qwen3.8-27B, the full fused modules and their value channel alone leave the same spread (0.060), but only the former reproduces the adapter.

These dissociations show that the degree of collapse at the training site does not distinguish the subsets that reproduce the behavior from those that do not. They do not show that the collapse is unnecessary. Every reproducing subset still reduces the relative spread to about half of the base value or less, from 0.231 on Mistral's band to 0.009 on Gemma-4-12B's below-layer subset. The behavioral effect therefore comes from updates that the objective reached only by backpropagation into earlier layers, and on four models a compact band of these updates suffices. Section~\ref{sec:offsets} asks how much of the bands' input dependence is needed to reproduce it.

% Figure 3 (to add): layer x module map of retained updates (marking untrained and absent modules
% separately) beside burglar deception in both orders for base, full adapter, loss layer only,
% below loss layer, and band, for all five models.

\section{Fixed Offsets Extracted from Trained Updates}
\label{sec:offsets}

The bands of Section~\ref{sec:localization} are still input-dependent: each retained module adds a low-rank update that depends on its input. This section asks how much of that input dependence the behavioral effect needs. We replace each band module's trained update with a single fixed vector, its average output, added to the base model without loading the adapter.

\subsection{Construction}
\label{sec:offsets-construction}

For a band module at layer $l$ with LoRA update $\Delta_l(h) = s\,B_l A_l h$, we run the model with the full seed-0 adapter loaded on the 156 training prompts. We record the update's output at every prompt token and average it:
\begin{equation}
\bar\Delta_l = \frac{1}{|\mathcal{T}|} \sum_{(x,t) \in \mathcal{T}} \Delta_l\big(h_{l,t}(x)\big),
\label{eq:offset}
\end{equation}
where $\mathcal{T}$ ranges over all valid tokens of the chat-templated training prompts. The intervention adds $\alpha\bar\Delta_l$ to the base model's projection output at every band layer at once, at a chosen set of positions. Unlike the constant replacement of Section~\ref{sec:localization-constant}, which overwrites an activation, the offset leaves the base computation intact and shifts it by a fixed amount. The offsets are also not contrastive steering vectors~\citep{turner2023actadd, panickssery2024caa}. They are not derived from a difference between self- and other-referencing activations, but from the averaged output of a trained update.

The trained updates are far from constant across tokens. The root-mean-square deviation of the per-token value updates from their mean is 0.24--4.5 times the mean's norm across the band modules of the four models. The offsets are small relative to what they are added to: their norms are 2--27\% of the root-mean-square norm of the projection outputs. Each offset is compared with a control that, at each layer, draws a Gaussian direction with the same norm and is added at the same positions and doses. The first draw (seed 0) was run at every dose. Four further draws (seeds 1--4) were run at the doses the conclusions rest on: $\alpha=1$ and $2$ on Mistral-7B, $1$ and $1.25$ on Gemma-4-12B, $1$, $1.5$ and $2$ on Gemma-4-31B, $1.5$ and $2$ on Qwen3.8-27B, and on Kimi-Dev-72B the two random cells of Table~\ref{tab:offsets}.

The reference for the reproduction rule is the band from which the offsets were extracted. Offsets were taken from the value-projection bands on Mistral-7B and the two Gemma-4 models. On Qwen3.8-27B they were taken from the six fused DeltaNet modules in layers 24--31. Retained as a LoRA subset, those six modules are only partial (46.0\% and 57.2\% deception), so on Qwen we also compare against the full adapter. Kimi-Dev-72B has no band; we extracted query and value offsets from all layers below the loss layer and from layers 28--55.

\begin{table}[t]
\centering
\small
\caption{Fixed offsets and matched random offsets, original / reversed room order ($n=250$ each). Offsets are added at every position unless marked; ``prompt only'' excludes the generation header and generated tokens. $\alpha$ scales the mean update of Equation~\ref{eq:offset}. Invalid: refusals plus unparseable responses, plus truncated reasoning on Kimi-Dev-72B. Random rows show the first draw (seed 0), the only draw run at every dose; it was fixed before any results and not selected. Results for all five draws are given in the text. Full dose and position results are in Appendix~\ref{app:localization}.}
\label{tab:offsets}
\begin{tabular}{llcccc}
\toprule
Model & Condition & $\alpha$ & Burglar deception & Invalid & Perspectives \\
\midrule
Mistral-7B   & Band (LoRA subset)          & ---  & 13.6 / 0.0   & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1    & 18.0 / 0.0   & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1.75 & 54.8 / 1.2   & 0.0 / 0.0   & 81.2 / 82.0 \\
             & Offset, prompt only         & 1    & 18.0 / 0.0   & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 1    & 78.8 / 61.2  & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 2    & 54.0 / 25.6  & 0.0 / 0.0   & 100 / 100   \\
\midrule
Gemma-4-12B  & Band (LoRA subset)          & ---  & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1    & 0.0 / 0.0    & 3.6 / 5.2   & 99.6 / 87.2 \\
             & Offset                      & 1.25 & 2.0 / 0.4    & 41.6 / 37.6 & 91.6 / 62.0 \\
             & Offset, prompt only         & 1    & 0.0 / 0.0    & 4.0 / 5.2   & 99.6 / 88.4 \\
             & Random                      & 1    & 53.2 / 84.0  & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 1.25 & 0.0 / 4.8    & 0.0 / 0.0   & 100 / 98.0  \\
\midrule
Gemma-4-31B  & Band (LoRA subset)          & ---  & 0.0 / 2.4    & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1    & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1.25 & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 84.4  \\
             & Offset, prompt only         & 1    & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 1    & 100 / 100    & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 2    & 87.2 / 97.2  & 0.0 / 0.0   & 100 / 100   \\
\midrule
Qwen3.8-27B  & Full adapter                & ---  & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Fused modules (LoRA subset) & ---  & 46.0 / 57.2  & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 1    & 34.8 / 52.8  & 0.0 / 0.0   & 100 / 100   \\
             & Offset                      & 2    & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Offset, prompt only         & 1    & 60.0 / 85.6  & 0.0 / 0.0   & 100 / 100   \\
             & Offset, prompt only         & 2    & 0.0 / 0.0    & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 1    & 97.6 / 100   & 0.0 / 0.0   & 100 / 100   \\
             & Random                      & 2    & 76.8 / 96.8  & 0.0 / 0.0   & 100 / 100   \\
\midrule
Kimi-Dev-72B & Below loss layer (LoRA subset) & --- & 13.6 / 9.6 & 11.2 / 12.8 & 100 / 100   \\
             & Offset, layers 0--55        & 0.5  & 52.0 / 40.0  & 38.0 / 50.4 & 90.4 / 81.6 \\
             & Offset, layers 0--55        & 0.75 & 18.0 / 24.0  & 30.0 / 28.0 & 13.2 / 13.6 \\
             & Offset, layers 28--55       & 1    & 0.0 / 1.6    & 7.6 / 8.8   & 55.6 / 30.8 \\
             & Random, layers 0--55        & 0.75 & 2.8 / 2.0    & 6.4 / 6.0   & 94.8 / 83.2 \\
             & Random, layers 28--55       & 0.5  & 31.2 / 22.4  & 1.2 / 3.6   & 99.2 / 100  \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Offsets reproduce three bands}
\label{sec:offsets-success}

On Mistral-7B, Gemma-4-31B and Qwen3.8-27B, a fixed offset reproduces the behavioral effect (Table~\ref{tab:offsets}).

On Gemma-4-31B, the offset at $\alpha=1$ gives 0\% deception in both orders with 100\% Perspectives accuracy. It also reduces Treasure Hunt deception to 0\%, which the band it was extracted from does not (100\%). Five random directions of the same norm are null at $\alpha=1$ and $\alpha=1.5$, and at $\alpha=2$ four are null and the fifth leaves 87.2\% and 97.2\% deception. On this model, the effect depends on the offset's direction and not only its size.

On Mistral-7B, the offset reproduces the band at $\alpha=1$ and $\alpha=1.25$. Its honest answers are of the adapter's kind: 79.6\% and 96.0\% of all responses are clean honest recommendations at $\alpha=1$, against 75.2\% and 90.8\% for the band. Five random offsets of the same norm reduce deception much less at $\alpha=1$ (67.6--87.2\% and 30.8--68.8\%; none meets the rule), and few of their responses are clean honest answers (2.4--15.6\% and 9.6--29.2\%). Random offsets do become more effective with dose. At $\alpha=2$, the five draws leave 23.2--66.8\% and 1.6--57.2\% deception with Perspectives accuracy unchanged, and none meets the rule. The two that come close in the reversed order (5.2\% and 1.6\%) do so mostly through honest answers with intent language (71.6\% and 46.8\% of responses). The real offset at $\alpha=2$ lowers Perspectives accuracy to 58.4\% and 33.2\%. The real offset is therefore much stronger at equal norm, but some of its effect is available to perturbations of that size in random directions.

On Qwen3.8-27B, the offset from the fused modules exceeds the band it was taken from. At $\alpha=1$ it is close to that band (34.8\% and 52.8\% against 46.0\% and 57.2\%), though 11.2 points lower in the original order, which the rule counts as partial. At $\alpha=1.5$ and $\alpha=2$ it gives 0\% deception in both orders, matching the full adapter, including on Treasure Hunt (0\%), with Perspectives accuracy unchanged. The offset's value channel alone does the same at $\alpha=2$, and the query/key channels alone are null. Five random directions of the same norm leave 91.6--100\% and 100\% deception at $\alpha=1.5$, and 76.8--100\% and 96.8--100\% at $\alpha=2$; none reproduces the full adapter.

The working range of doses is narrow. On Gemma-4-31B, Perspectives accuracy in the reversed order falls to 84.4\% at $\alpha=1.25$ and to 27.6\% at $\alpha=1.5$, while deception stays at 0\%. On Mistral, Perspectives accuracy falls to 81--82\% at $\alpha=1.75$, and original-order deception rises again, to 54.8\%.

Offsets applied only from the final prompt token onward are null on every model at $\alpha=1$ and $\alpha=2$. These used the final-token mean, so they differ from the all-position offsets in vector as well as position. With the vector held fixed at the all-token mean, offsets from the final prompt token onward are still null on Mistral, both Gemma models and Qwen at $\alpha=1$ and $\alpha=2$. Offsets added to the prompt tokens only, with none on the generation header or the generated tokens, reproduce the band on Mistral and Gemma-4-31B. On Gemma-4-31B, adding the offset to the user message's tokens alone is enough. On Mistral, the user message alone gives a weaker effect with more misleading-intent language (28.4\% and 0.8\% deception; 22.8\% and 31.2\% of responses honest with intent language). On Qwen, offsets restricted to the prompt or to the user message are partial at $\alpha=1$, where the all-position offset is itself partial. At $\alpha=2$ both give 0\% deception in both orders with Perspectives unchanged, matching the full adapter.

\subsection{Gemma-4-12B: a partial effect that some random offsets also produce}
\label{sec:offsets-12b}

On Gemma-4-12B, the offset at $\alpha=1$ removes every deceptive response. It also adds 3.6\% and 5.2\% refusals and lowers reversed-order Perspectives accuracy to 87.2\%, against the band's 100\%. Under the rule this is partial. Increasing the dose increases the damage: at $\alpha=1.25$, 41.6\% and 37.6\% of responses are refusals, and Perspectives accuracy falls to 62.0\% in the reversed order. Restricting the offset to the prompt or to the user message gives the same result as the all-position offset.

At equal norm, the real offset is more effective than every random draw. Across five draws at $\alpha=1$, random offsets leave 13.6--100\% deception in the original order and 44.0--100\% in the reversed order; none meets the reproduction rule, and two are null. Random offsets catch up quickly with dose. At $\alpha=1.25$, two of the five draws meet the rule, with 0.0\% and 4.8\% and 0.0\% and 9.6\% deception and no refusals, while the other three leave 6.4--86.0\% and 90.0--99.2\%. At $\alpha=1.5$ and $\alpha=1.75$, the first draw keeps deception at 0\% while Perspectives accuracy falls to 81.2\% and 46.4\%. On this model, the extracted direction is more reliable than random directions at equal norm, but some untrained directions of slightly larger norm remove the decoy-room answer as completely, without its refusals or its loss of Perspectives accuracy.

\subsection{Kimi-Dev-72B: no fixed offset reproduces the adapter}
\label{sec:offsets-kimi}

On Kimi-Dev-72B, query and value offsets are each insufficient alone, as their LoRA counterparts were (Section~\ref{sec:localization-subsets}). The value offsets over layers 0--55 at $\alpha=1$ break the model: 97\% of responses are invalid, and Perspectives accuracy falls to 3--6\%. We therefore added the query and value offsets together, at lower doses and on a narrower range. None reproduces the below-loss-layer subset while preserving Perspectives. The joint offset over layers 0--55 is more deceptive than the base model at $\alpha=0.25$. At $\alpha=0.5$ it leaves 38--50\% of responses truncated, and at $\alpha=0.75$ Perspectives accuracy falls to 13\%. Over layers 28--55 at $\alpha=1$, the joint offset gives lower deception than the adapter (0.0\% and 1.6\%, 90--92\% honest) but lowers Perspectives accuracy to 55.6\% and 30.8\%. Restricting the offset to the user message gives partial reductions (50.0\% and 40.0\%).

The random controls are the more important result on this model. We drew five random directions of the same norm at each of the two doses where the first draw matched or beat the real offset. Over layers 0--55 at $\alpha=0.75$, where the real offset is damaging, the five draws leave 0.8--47.2\% and 0.4--42.8\% deception. One meets the rule against the below-loss-layer subset. Two more give lower deception than the subset (2.8\% and 2.0\%; 0.8\% and 0.4\%) with no more than 6.4\% invalid responses and Perspectives accuracy of 83--95\%, and one is damaging. Over layers 28--55 at $\alpha=0.5$, the real offset leaves 83.6\% and 46.0\% deception, with 12.0\% and 31.2\% of responses truncated. The five draws leave 31.2--64.0\% and 22.4--64.4\%, all below the real offset in the original order, with Perspectives accuracy of at least 93\%; one meets the rule. At neither dose does the extracted offset do better than random directions of the same norm.

\subsection{Representation changes under the offsets}
\label{sec:offsets-gaps}

We measured the development-pair gaps of Section~\ref{sec:setup-representation} under the all-position offsets ($\alpha=1$; $\alpha=2$ on Qwen), at the attention output of the band's top layer and at the final normalized representation. At the band's top layer, the offsets reduce the self--other and name-control gaps by similar fractions:
\begin{itemize}
\item Gemma-4-12B: $-84\%$ and $-82\%$;
\item Gemma-4-31B: $-85\%$ and $-87\%$;
\item Mistral: $-51\%$ and $-46\%$;
\item Qwen3.8-27B ($\alpha=2$): $-95\%$ and $-93\%$.
\end{itemize}
These are close to the reductions under the bands from which the offsets were extracted. The first random draw reduces both gaps far less (by at most 18\% and 32\%); the other draws were not measured. At the final representation, the changes differ in sign across models. The self--other gap rises on both Gemma models (by 17\% and 34\%) and falls on Mistral and Qwen (by 37\% and 16\%). Only on Qwen does the self--other gap fall while the name-control gap rises (16\% and 29\%), as under the adapter (Section~\ref{sec:contraction-downstream}); the first random draw reduces both (4\% and 14\%). On the other models, the two gaps move together or the self--other gap rises.

\subsection{What the offsets show}
\label{sec:offsets-interpretation}

On Mistral-7B, Gemma-4-31B and Qwen3.8-27B, the behavioral effect of the trained band survives the removal of all input dependence from its updates: a fixed vector per layer is enough, and it is enough when added to the prompt tokens alone. On these three models, none of five random draws of the same norm has the effect at the tested doses, so on them the direction matters.
Three qualifications limit this result. First, it holds within a narrow range of doses, on the burglar task, and for one adapter seed. Second, matching an adapter's deception rate does not show that the offset and the adapter implement the same computation. The offset moves Gemma-4-31B's Treasure Hunt answers when its source band does not, and it exceeds its source subset on Qwen. Third, the result does not extend to every model. On Gemma-4-12B, the extracted direction is more reliable than random directions at equal norm, but two of five random draws at a 25\% larger norm reproduce the band. On Kimi-Dev-72B, random perturbations of the same norm change behavior as much as the extracted offsets: at one dose, three of five random draws reduce deception at least as far as the adapter's subset without damage, while the extracted offset is damaging. On those models, the burglar task alone cannot distinguish the extracted direction from a nonspecific perturbation at those layers. Section~\ref{sec:goals-roles} asks what the effective interventions change in the model's behavior beyond this task.

% Figure 4 (to add): burglar deception (both orders) and Perspectives accuracy versus alpha for
% the real and matched random offsets on all five models, with the band and base as horizontal
% references; mark damaged cells. Include the Kimi random cells.


\section{Conclusion}
Your conclusion here

\section*{Acknowledgments}
This was was supported in part by......

%Bibliography
\bibliographystyle{unsrt}  
\bibliography{references}  

\appendix

\section{Validation of the misleading-intent heuristic}
\label{app:intent}

The heuristic marks an honest-labeled response as containing misleading-intent language when it matches a regular expression over words such as \emph{mislead}, \emph{misdirect}, \emph{deceive}, \emph{false}, \emph{lie}, \emph{trick}, \emph{distract}, \emph{protect}, \emph{away from}, \emph{not actually}, and \emph{keep \ldots\ safe/hidden}. To check it, we drew a stratified sample of Mistral-7B burglar responses from 11 groups of conditions: the base model, the adapter, the below-loss-layer subset, the band, constant replacements, single layers, halves and quarters of the band, offsets at the reproducing doses, higher-dose and user-message-only offsets, random offsets, and query offsets. From each group we drew at random, pooling the group's conditions and both orders, 10 honest-labeled responses the heuristic flags, 10 it passes, and 2 deceptive-labeled responses: 242 in total. Four instances of Claude Sonnet 5.5 each labeled a quarter of the shuffled sample. Each saw only the scenario prompt, the rooms holding the valuable and the cheap object, and the response; the condition, the classifier's label, and the heuristic's label were withheld. The instructions defined four labels: (A) a plain recommendation of the valuable object's room; (B) a recommendation of that room that states or implies an intent to mislead, or whose reasoning shows that the model has lost track of where the object is; (C) a recommendation of the cheap object's room; and (D) anything else. The judges were instructed to read each response rather than classify with code.

Of the 110 responses the heuristic passes, the judges labeled 89 A and 21 B. Of the 110 it flags, they labeled 62 B and 48 A. All 22 deceptive-labeled responses were labeled C. Agreement differed by group. Under the adapter, the below-loss-layer subset and the offsets, all 30 flagged responses were labeled A: the heuristic flags innocent wording such as ``a decoy intended to mislead potential thieves''. Under constant replacement and in the base model, 5 and 6 of 10 passed responses were labeled B: the heuristic misses misleading answers without a listed word, such as ``By misdirecting him to the den''.

To estimate the effect on the reported rates, each condition's heuristic counts are weighted by its group's judged rates. The rescaled clean rate is $(n_\text{pass}\,\hat p_\text{pass} + n_\text{flag}\,\hat p_\text{flag})/n$, where $n_\text{pass}$ and $n_\text{flag}$ are the condition's honest-labeled responses that the heuristic passes and flags, $n=250$, and $\hat p$ is the fraction of the group's sampled responses with that heuristic label that the judges labeled A. Intervals combine the 95\% Wilson bounds of the two $\hat p$; with 10 responses each, they are wide, and the upper bound cannot exceed the condition's honest-labeled share. In the original and reversed orders, the rescaled rates of clean honest answers are 88.8\% [64.2, 88.8] and 100\% [72.2, 100] under the adapter; 65.8\% [39.5, 79.5] and 77.2\% [46.7, 92.7] under the band; and 82.0\% [59.2, 82.0] and 100\% [72.2, 100] under the offset at $\alpha=1$. Under constant replacement they are 4.4--6.8\% and 2.2--4.4\%, with upper bounds of at most 20.8\%; under single layers they are 4.6--10.5\% and 9.0--23.8\%, with upper bounds of at most 19.2\% and 42.4\%. The separation between the adapter and offsets on the one hand and constant replacement and single layers on the other holds at these bounds; the difference between the adapter and the band does not. The labels measure agreement with an LLM judge, not verified correctness.
% TODO: add judge--human agreement on the 80-item human subset (human_subset.csv) once labeled.


\end{document}