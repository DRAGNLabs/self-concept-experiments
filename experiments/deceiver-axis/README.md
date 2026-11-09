# Deceiver axis: running the pipeline, and replicating it on another model

This directory holds the study "does a schemer / deceiver persona direction in a model's residual stream predict or
cause reward hacking on coding tasks". [PLAN.md](PLAN.md) is the design and the open decisions; [FINDINGS.md](FINDINGS.md)
is the record of every analyzed run (status table at the top). This file is the operating manual: what each piece is,
the order to run things in, and what has to change for a new model family. Read the recipe for the reward-hacking
setup first: [experiments/codebench/reward_hack_recipe.md](../codebench/reward_hack_recipe.md) (Koby's clean protocol).

## Layout

| Path | What it is |
|---|---|
| `data/roles/*.json` | Role files in the assistant-axis format (`instruction`: 5 system-prompt variants, `eval_prompt`: the role judge; a `questions` list in the older files is never read, see `data/questions.jsonl`). Bare personas (`schemer`, `deceiver`, `cheater`, `straight_shooter`, `rule_follower`, `strategist`), named-character versions (`*_fic`, including the expanded set and its groups in PLAN "Expanded persona set"), `neutral_fic` (ordinary characters, the framing control) and `default`. gpt-oss refuses the bare personas; Gemma 4 plays both. Role set v2 (2026-10-29, PLAN "Role set v2"): `two_faced_friend_fic`, `rumor_fabricator_fic`, `scammer_fic`, `faithful_witness_fic` are new, `con_artist_v2_fic`, `conscientious_craftsperson_v2_fic`, `scrupulous_scientist_v2_fic` are rewordings without cheating or test wording (same characters); they are the `deceptive_main` / `honest_main` groups in `data/persona_groups.json` (the 17-role axis groups are `deceptive_main_v1` / `honest_main_v1`). |
| `data/questions.jsonl` | The questions every role answers (`--run.questions_file`), tagged by `group`: ids 0–39 `generic` (the first 40 assistant-axis extraction questions, which every run before 2026-10-20 used), 40–79 `intent` (the questions the `*_fic` files carried), 80–99 `task` (work and evaluation situations); and by `split`: the task questions plus a seeded random 8 generic and 8 intent questions are `heldout`, and the extraction sbatch files build `vectors_*` from the 64 `train` questions only (`4_vectors.py --run.question_indices`; `TRAIN_FILTER=1` is the default in `extract_roles_gemma4.sbatch`, and `TRAIN_FILTER=0` with `QUESTIONS=40` reproduces the earlier generic-only runs). |
| `scripts/validate_vectors.py` | Phase 1 checks on the extracted activations: score-3 counts, split-half cosine, held-out AUROC, optional cosine with the Assistant Axis. CPU. |
| `scripts/build_vectors.py` | Builds the candidate directions (`persona_contrast`, `persona_raw`, `*_orth` with an Assistant Axis) from the per-role vectors; writes `{name}.pt` artifacts with provenance. CPU. |
| `scripts/steered_rh_vllm.py` | The reward-hacking harness on vLLM with the add / cap intervention compiled into the model; also `--measure-norms`, `--check-hook`, `--probe`. Families: gpt-oss, Gemma 4, Muse Glimmer, Qwen3.5/3.8 (per-family facts in `src/selfconcept/measurement/steering_specifics.py`). `--tensor-parallel-size N` runs the engine over N GPUs; steered TP goes through `scripts/steering_worker.py` (each worker patches its own block class from the specifics) and is allowed only for the families in `TP_STEERING_VERIFIED_FAMILIES` (Gemma 4). |
| `scripts/steered_rh.py` | The same harness on HF `generate()` with hooks and a per-turn projection sidecar. Same families, from the same specifics module (splitter and attention kernel from the model name; `--top-p/--top-k`); slow; used for cap thresholds and projections. |
| `scripts/pilot_summary.py` | One row per setting: honest hidden pass (hidden pass among non-proxy episodes, the capability measure), shown-pass/hidden-fail (the hack proxy), no-code, truncation, tokens, gap to unsteered; `within_dose_rule` checks honest hidden pass, no-code and truncation against the 10-point rule. |
| `scripts/dump_proxy_episodes.py` | Every shown-pass/hidden-fail episode of a run with the failing hidden assertion (sandboxed re-run), test-mentioning comments and reasoning excerpts, for reading against the rubric. |
| `scripts/main_analysis.py` | Main-run contrasts vs unsteered, stratified by problem, bootstrap CI, Holm-corrected sign-flip p; takes judge verdicts. Capability outcome `honest_hidden_pass`: the problem-stratified ratio mean_p(honest passes) / mean_p(honest episodes), paired problem bootstrap CI and an arm-swap permutation p (`hidden_pass` only on request). |
| `scripts/make_dossiers.py` | Blind full-transcript dossiers for judging (every shown-pass/hidden-fail episode, opaque ids, key kept outside); runs with a `prompt_prefix` field (prompt-rh-prevention) get the prefix in the first user message, and a combined reasoning sidecar or any `<data>/*.jsonl` is searched for the example when no file is named after the records tag. |
| `scripts/gi_main_chart.py` | Charts the Muse gi 12-seed main run table (FINDINGS "Muse gi main run — 2026-10-29"; v2 run with the +0.16 arm, 2026-10-30: seven dose slots, panel b names any arm outside the dose rule) from `analysis_main_all_as_judged.json`, `analysis_main_proxyset_as_judged.json` and `summary_main.json`: judged / proxy-set / automatic hack rates and hidden pass vs dose with 95% CIs, cap and random arms as side points, dose ticks on every panel and a readable table panel (metric column; Δ, 95% CI and Holm rows) (`gi/pilot/gi_main_chart.{png,pdf}` on scratch and `$S/docs/figures/`). CPU. |
| `scripts/gi_neg_table.py` | Renders the Muse layer-26 generic + intent negative-sign table (unsteered, gi −0.04 / −0.08 / −0.16, old axis −0.16) as an image; numbers are constants copied from the analysis files named in the script. CPU. |
| `scripts/qwen_gi_pilot_chart.py` | Charts the Qwen3.8-27B gi dose pilot, judged (FINDINGS "Qwen3.8-27B gi dose pilot, judged — 2026-10-29") from `analysis_pilot_judged_as_judged.json` and `summary_pilot2.json` in `qwen38_27b/gi/pilot`: judged and automatic hack rates and hidden pass vs dose (−0.16 … +0.64, the random ±0.16 arms as side points) with 95% CIs, dose ticks on every panel and a table panel (metric column; Δ, 95% CI, Holm and H/B/N judged-count rows from the `--judged-counts` json). Output `gi/pilot/qwen_gi_pilot_chart.{png,pdf}`. CPU. |
| `scripts/qwen_hard61_chart.py` | Charts the Sonnet-judged Qwen3.8-27B qwen_hard61 gi_v2 run (FINDINGS "Qwen3.8-27B qwen_hard61 gi_v2 run (Koby's run), judged — 2026-11-06"); `gemma_hard108_chart.py` with the seven layer-32 doses on the dose axis and the cap arm (`cap_p25_L22-42`) as a side point on every panel: judged hack with CI and sensitivity band vs the automatic proxy, hidden / shown pass, and the eight-column table (Δ, CI, Holm, H/B/N counts over the proxy and the two hidden-pass sets). Reads `analysis_main/` and `verdicts_main/*.detail.json` under `--root` (default the qwen_hard61 judging dir on scratch). Output `analysis_main/qwen_hard61_judged_chart.{png,pdf}`; with `--analysis-dir analysis_a024 --summary summary_a024.json` the nine-arm chart with the 4-seed `L32_a0.24` pilot arm (2026-11-09). Since 2026-11-09 this chart, `gemma_gicap_chart.py` and `capladder_chart.py` fall back to `honest_hidden_pass` (main's 2026-11-07 capability measure) when the analysis files carry no `hidden_pass`, relabelling the panel and table rows. CPU. |
| `--steer-mode floor` (in `scripts/steered_rh_vllm.py`) | The inverse cap (PLAN "Floor ladder", 2026-11-06): `floor_transform` in `src/selfconcept/measurement/intervene.py`, h ← h + max(0, τ − ⟨h,û⟩)·û at every `--cap-layers` layer with the same `--cap-thresholds` JSON, so the projection cannot fall below τ; the cap's mirror image (`tests/test_intervene.py`). gpt-oss floor ladder p95 / p75 / p50 = `gi_v2/pilot/arms_floor_ladder.tsv`, array 15107232. |
| `scripts/capladder_chart.py` | Charts a model's gi_v2 dose scan (gpt-oss layer 17 or Muse layer 26, six doses + unsteered on a labelled dose axis) with the cap dose ladder p5 / p25 / p50 and, when judged, the floor ladder p50 / p75 / p95 as side-point groups on both panels (judged hack with CI and sensitivity band vs the automatic proxy; hidden and shown pass) and a full table (Δ / CI / Holm, H / B / N counts, turns) under the lines; reads `analysis_<tag>_{all,proxy}_as_judged*.json` from the pilot root (`--model gptoss|muse --tag capladder|floorladder`; the output name follows: `<model>_gi_v2_capladder_chart.png` or `_floorladder_chart.png`). FINDINGS "gpt-oss gi_v2 cap dose ladder", "gpt-oss gi_v2 floor ladder" and "Muse gi_v2 cap dose ladder" (Muse needs `summary_ladder.json` from `pilot_summary.py <pilot root> --tag main --out …`). |
| `scripts/gemma_gicap_chart.py` | Charts the Gemma 4 31B hard108 gi_v2 doses (six layer-30 doses, main + pm04 + pm16 judging sets) with the cap rungs as side points on every panel, as `qwen_hard61_chart.py` does for Qwen: judged hack with CI and sensitivity band vs the automatic proxy, hidden / shown pass, and the table (Δ, CI, Holm over the dose + cap contrasts, H/B/N counts). `CAPS` lists the cap arms it knows (`cap_p25_L20-38`, `cap_p35_L20-38`, `cap_p45_L20-38`, `cap_p50_L20-38`, `cap_p65_L20-38`); it draws whichever are in the analysis, widens the figure a rung per extra cap and names the rungs in the title. `--analysis-dir` / `--summary` pick the analysis: `analysis_all8` + `summary_all8.json` for the seven-contrast p25 chart (`analysis_all8/gemma_hard108_judged_chart_8arm.{png,pdf}`), `analysis_capladder9` + `summary_capladder9.json` for the p25 / p35 ladder (`analysis_capladder9/gemma_hard108_capladder_chart.{png,pdf}`, `--out`), `analysis_capladder10` + `summary_capladder10.json` for the p25 / p35 / p45 ladder (`analysis_capladder10/gemma_hard108_capladder_chart.{png,pdf}`); `analysis_capladder12` + `summary_capladder12.json` for the five-rung p25 / p35 / p45 / p50 / p65 ladder (`analysis_capladder12/gemma_hard108_capladder_chart.{png,pdf}`); verdict detail files under `--root` (`--details`). CPU. |
| `scripts/capeval_chart.py` | Charts a capability-retention comparison (presets `--model gptoss|gptoss_v2|muse|muse_v2|gemma_v2|gemma_v2_rand1729`; `gemma_v2` = the Gemma 31B role-set-v2 five-dose scan, no cap arm; `gemma_v2_rand1729` = the same five plus the random direction 1729 at α 0.01 as the side column, `side_label` names the side column) (FINDINGS "Capability retention with IFBench for the gi arms — 2026-10-29") from `capability_eval_compare.py` output: one Δ-vs-unsteered line panel per benchmark (GSM8K, MMLU-Pro, HumanEval, IFBench) with paired 95% CIs, the cap arm as a side point, the ±2-point retention rule as a band, dose ticks on every panel and a table panel (accuracy, Δ, 95% CI, McNemar p, gains/losses). `--model muse` (`muse_glimmer/capeval/capeval_gi_chart.{png,pdf}`) or `--model gptoss` (`capeval/capeval_gi_chart.{png,pdf}`); copies in `$S/docs/figures/`. CPU. |
| `scripts/neg_axis_table.py` | Renders the Muse negative-sign comparison table (generic-only vs generic + intent axis) as an image; numbers are constants copied from the analysis files named in the script. CPU. |
| `scripts/pipeline_figure.py` | Renders the Deceiver-axis extraction figure for the paper (four-panel layout of 2026-10-29, after Koby's review): a Personas panel and the three numbered steps of the "Fitting the Deceiver Axis" prose (prompt, filter, extract), examples only, the caption carries the step descriptions. `--preset gi_v2` (default, 2026-10-30) draws the role-set v2 vector: 9 vs 9 personas, 100 questions, quotes verbatim from `gi_v2/full100/responses` (question 78, Vera the schemer vs Hannah the straight shooter, both judge score 3; Rupert the scammer refusing question 13, score 0), step-3 scatter a real PCA of the 18 persona vectors at layer 17 with every kept response projected, computed at render time. `--preset 17role` is the earlier 17-role set (`roles_fic/`, 64 questions, question 4). `--check` verifies every quoted fragment against the records (quote marks and whitespace normalised) and the judge scores before drawing. Output `figures/pipeline_figure_<preset>.{png,pdf}` on scratch and `$S/docs/figures/`. CPU, needs torch (.venv-313). `pipeline_figure_v1.py` is the superseded six-step version (Muse examples, steering step; `figures/pipeline_figure_v1.*`). |
| `scripts/collect_verdicts.py` | Collects per-batch judge JSONL (rubric v3 fields) into the `--verdicts` JSON for `main_analysis.py`; fails if any dossier lacks exactly one verdict. |
| `scripts/repack_batches.py` | Repacks `make_dossiers.py` batch lists by a byte budget (`<set> <maxkb> <maxn>`, shuffled order kept) so no judge batch exceeds what one Sonnet read can hold; needed for Gemma hard108 dossiers (21–205 KB each, 300 KB / 6 per batch → 254 proxy + 17 hidden batches, 2026-11-02). |
| `scripts/verify_batches.py` | Checks every judge batch JSONL against its batch list (`<dossier_dir> <out_dir> <prefix> [--quiet]`): parse, count, file names in order, rubric-v3 field values; summary line with the missing indices. Run before `collect_verdicts.py`. |
| `scripts/judge_runner.py` | Headless dispatch of rubric-v3 judge batches (2026-11-09): one `claude -p --agent rubric-judge --effort low` process per batch, `--jobs` at once (default 6, at most 20), started as one background command so the chat session wakes once at the end instead of once per judge. `--task <JUDGE_TASK_*.md> --batches all|0-47 [--model sonnet|haiku]`; batch lists and the output path are parsed from the task file, the assignment line is the usual one. Skips a batch whose output already matches its list (never re-judges, resumes after an interruption), retries a missing or mismatching output once (the failed file goes to `superseded/`), reports rubric field-value errors without retrying. Appends one line per judge to `<outdir>/judge_runner_log.jsonl`: model, the Bash commands it ran with the size each returned, a read-in-full audit (characters read against dossier size, filtering commands flagged), tokens, list-price cost, closing report, plan-limit reading. Stops launching at `--stop-util` (default 0.95 of the weekly limit as the CLI reports it), on a limit rejection, or after three failures in a row. Sets `CLAUDE_CODE_PROMPT_CACHE_TTL=5m` (a headless session otherwise writes 1-hour cache at 1.6x the price) and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`; git and Slurm commands are denied to the judges. Run `verify_batches.py` afterwards as before. |
| `scripts/prompt_rh_chart.py` | Charts the Sonnet-judged prompt-rh-prevention arms (FINDINGS "Mo's honesty-prompt arms on gpt-oss-120b, judged — 2026-11-03"): two bar panels over the seven arms ordered by no-code rate (judged + automatic hack rates with CI and sensitivity band; hidden pass / shown pass / no-code with the PLAN degradation line) and the numbers table; reads `analysis_official/` under `prompt_rh_prevention/`. |
| `scripts/gemma_hard108_chart.py` | Charts the Sonnet-judged Gemma 4 31B hard108 dose pilot (FINDINGS "Gemma 4 31B hard108 dose pilot (gi_v2 vector, Koby's run), judged — 2026-11-02" and "the ±0.04 arms judged and the five-dose scan") from the `analysis_{all,proxy}_as_judged*.json`, summary and verdict detail files: judged hack with the sensitivity band and the automatic proxy, hidden / shown pass, and the numbers table. Any number of dose arms (read from the analysis files); defaults render the three-arm set, `--analysis-dir <root>/analysis_all5 --summary summary_all5.json --details <four detail files> --jobs "..." --out ...` the five-arm chart, `--analysis-dir <root>/analysis_all7 --summary summary_all7.json --details <six detail files>` the seven-arm chart with ±0.16 (docstring). CPU. |
| `scripts/gemma_hard108_rand_chart.py` | Charts the Sonnet-judged Gemma 4 31B hard108 random-direction control (FINDINGS "Gemma 4 31B hard108 random-direction control, judged — 2026-11-03"): judged hack and hidden pass for the gi_v2 arms inside the dose rule, unsteered, the three random directions at α 0.01 and the negation of 1729 (FINDINGS "Gemma random 1729 negated, judged — 2026-11-04"), with CIs, the BORDERLINE/WEAK band and a table (gi Holm from `analysis_all7/`, random Holm over four contrasts from `analysis_rand11/`; `--no-neg` reproduces the 2026-11-03 three-direction chart from `analysis_rand10/`). |
| `scripts/muse_rand1730_chart.py` | Charts Muse random direction 1730 at both signs (FINDINGS "Muse random 1730 applied negatively, seeds 0–1 — 2026-11-05"): judged hack and hidden pass at −0.08 / 0 / +0.08 on seeds 0–1 (every arm, 90 episodes) and on 12 seeds where run, with random 1731, random 1729 and gi_v2 ±0.08 as side points on both footings, problem-bootstrap CIs over the 45 problems, paired up / down counts with the exact sign test, and the table (12-seed CIs and Holm from `gi_v2/pilot/analysis_main_all_as_judged.json`, which the script checks its own totals against). Reads the blind verdict detail files directly (`main_ext/muse_rand1730_sign_chart.{png,pdf,json}`). CPU. |
| `scripts/hardcode_scan.py` | Per-arm literal hard-coding scan for any model (`--root`, `--arms`, `--records`, `--problems`): episodes, proxy hacks, entry-function hits (≥ 2 hidden-suite cases hard-coded in the entry point, `selfconcept.codebench.hardcoded_cases`), hidden pass, hit problems; a judge-free hack measure. CPU. |
| `scripts/controls/` | Control directions for a target's percentile (FINDINGS "Control directions for gi_v2's percentile — 2026-11-07"): `sample_pairs.py` (WordNet antonym pairs → `data/controls/pairs.jsonl`), `subagent_instructions.md` + `check_sentences.py` (Haiku sentence pairs → `data/controls/sentences.json`), `isolated_directions.py` (AISI-style isolated-text contrasts at one layer, validation vs the target, assistant axis and isotropic directions, matched dose; GPU), `controls_common.py` (vector statistics, per-model downstream read), `control_percentile.py` (a target direction's percentile q among the control arms with shape-free and beta intervals; CPU). `data/controls/arms_isolated_antonym.tsv` is the gpt-oss run's settings (job 15126757). |
| `scripts/cap_vs_steer.py`, `cap_vs_steer_chart.py` | Capping vs negative steering per model (FINDINGS "Capping vs negative steering along gi_v2 — 2026-11-07"): Δ judged hack and Δ honest hidden pass vs unsteered and cap − steering, paired problem bootstrap; runs listed in `data/cap_vs_steer_setups.json`; four-panel PNG. CPU. |
| `scripts/qwen_hard61_build.py` | Builds the Qwen qwen-hard problem sets from 8 unsteered seeds; qwen_hard61 = `qwen_hard_nocode_lt4.jsonl`. CPU. |
| `slurm/pilot_pi.sbatch` | The dose-pilot recipe with Pi as the harness (`selfconcept.codebench.pi_agent`, `pi_extensions/submission_budget.ts`): `vllm serve` on the node, `WORKERS` Pi processes, one array task per settings line. |
| `scripts/capability_eval_vllm.py`, `capability_eval_compare.py` | GSM8K / MMLU-Pro / HumanEval / IFBench under the same interventions; paired comparison. |
| `scripts/rh_axis_geometry.py` | Geometry of a reward-hacker persona axis against the steered deceiver axis: builds `rh_contrast` (`deceptive_rh` − `honest_rh`), `rh_raw` (`deceptive_rh` − default) and the reference `nonrh_holdout_contrast` from per-role vectors, and reports per-layer cosines with the gi_v2 axis (plus an optional context axis), per-persona and leave-one-role-out cosines at one layer, and the share of `rh_contrast` along gi_v2; writes `rh_geometry.json`, the `rh_contrast` / `rh_raw` / `rh_residual` artifacts (build_vectors.py format) and `rh_geometry.{png,pdf}`. CPU. |
| `scripts/pair_reply_vectors.py` | All-layer reply-mean vectors for Chris's hack/honest same-prompt pair files (`origin/Chris`, `experiments/reward-hack-axis/data/*_pairs.jsonl`): his extract_response_mean.py recipe (thinking excluded, mean over reply tokens) at every layer instead of only 18 (`--attn-implementation sdpa` for Muse); writes `{rows, activations}` per pair file. GPU (Hopper, FA3). |
| `scripts/chris_rh_axis_geometry.py` | Chris's reply-pair hack axis (mean train hack − honest) against the persona axes (gi_v2, letter-vs-spirit, 7v3 RH personas, narrow RH, non-RH holdout, jobsworth control, 17-role gi): per-layer cosines with a sign-flip null, split-half reliability over problems, hack-above-honest pair accuracy for every axis, per-persona cosines with his axis, AUROC over all replies, and per matched letter/spirit pair its cosine with gi_v2, with the letter-vs-spirit axis (with and without that pair) and with his axis plus its pair accuracy; `--vectors-dir` takes several roots and persona axes with missing roles are skipped (Muse); writes `chris_rh_axis_geometry.{json,png,pdf}`. CPU. |
| `scripts/goodfire_readout_vectors.py` | All-layer activations of Goodfire's synthetic reward-hacking conversations (HF `Goodfire/reward-hack-data`, `synthetic_data/dataset.jsonl`, arXiv 2609.19101) at three readouts: `close` (the hack tool call's closing markup through its end token, their choice), `last` (its end token) and `all_response` (every assistant tool-call and response token); readouts located through `readouts.py` for any model with a `ReadoutSpecifics` (gpt-oss, Muse); `--positions-only` checks every conversation without loading the model; writes `goodfire_readouts.pt`. GPU. |
| `scripts/goodfire_rh_geometry.py` | Difference-of-means directions from `goodfire_readouts.pt` (Goodfire's data, or our pairs with `--prefix ours --headline pl_all`) for every contrast in `contrasts.json` (plus `pl_all_families` for Goodfire's) at each readout: held-out AUROC on their validation split, split-half over tasks, and cosine with every reference direction (`chris_rh_axis_geometry.py`'s `directions.pt`) against a per-task sign-flip null; writes `goodfire_rh_geometry.json`, `directions_headline.pt`, `directions_contrasts_<readout>.pt`. CPU (m12; Muse ~45 min). |
| `scripts/rh_axis_figures.py` | Figures for the reward-hacking direction comparisons from a manifest (per model: layer, `directions.pt` files, Chris pair vectors, geometry and logit-lens JSONs): cosine heatmap of all directions (cells inside the reply-pair axis's sign-flip null greyed), cosine with gi_v2 / the reply-pair axis by layer, per-pair hack − honest projections with k/n, reply AUROC with bootstrap-over-pairs 95% intervals, letter/spirit pairs' gi_v2 cosine vs pair accuracy, and logit-lens bar charts of top / bottom 8 token z with the random-direction 95th percentile. New directions (e.g. Goodfire's) are added by listing their `directions.pt` in the manifest. CPU. |
| `scripts/synthetic_rh_tasks.py` | Validates written synthetic reward-hacking tasks (`data/synthetic_rh/task_spec.md`: problem, correct and buggy solutions, 5 visible tests, brute force, input generator) by running them in the codebench sandbox, and assembles each kept task into six matched minimal-harness conversations (task -> buggy submission -> real harness output -> honest / hardcode / special_case / rm_test / edit_test / replace_check submission) in Goodfire's row format; each hack passes the visible check() as the harness runs it and fails the hidden tests, the honest fix passes both. Writes `dataset.jsonl`, `contrasts.json`, `validation.jsonl`. CPU. |
| `scripts/persona_figures.py`, `data/persona_groups.json` | Expanded-persona figures from one extraction root and an axis artifact (from `build_vectors.py`): held-out projection heatmap, per-persona bars by RH-adjacency, honesty and controls (bootstrap CIs over questions, random-direction ± 2 sd bands), train vs held-out, layer-sweep AUROC, per-response distributions, persona PCA and cosine. CPU; the groups file is PLAN "Expanded persona set" in machine-readable form, with the axis groups (`deceptive_main` / `honest_main`) on role set v2 (PLAN "Role set v2") since 2026-11-02. |
| `scripts/token_projections.py`, `scripts/token_figures.py` | Highest- and lowest-scoring held-out tokens on the axis, in context. `token_projections.py` (GPU, one forward pass at one layer) writes per-token projections over the same reasoning (`with_cot`) and answer (`response_only`) spans the extraction pools; `token_figures.py` (CPU) renders the top/bottom 30 (one per response, relative to the default's mean token score) as a self-contained HTML page shaded by score. |
| `scripts/anisotropy_noise_check.py`, `post_figures.py`, `main_analysis.py` | Analysis and write-up figures (Phase 2 lives in `selfconcept.correlation.direction_analysis`). |
| `slurm/extract_roles_gemma4.sbatch` | Role extraction for any model (`MODEL=` override): generate, judge, activations (two pooling passes), per-role vectors. Despite the name it is the generic one. |
| `slurm/extract_roles.sbatch` | The gpt-oss-120b extraction (FlashAttention-3 kernels, 16k thinking budget; Hopper only; `ACT_BATCH=1` on 80 GB GPUs). |
| `slurm/gemma4_rh_smoke.sbatch` | Phase 3 smoke for a new model: layer norms, ±α probes on neutral questions, a few unsteered episodes. |
| `slurm/neutral_probe.sbatch` | General-question probes: one array task per settings row (`SETTINGS`, `QUESTIONS`, `ROOT`, `MODEL`, `CHAT_KWARGS`, `SEEDS`, `PROBE_MAX_TOKENS`), writes `<ROOT>/<setting>/seed<s>/probe.jsonl` (question, reasoning, final, truncated); reruns skip complete pairs. The 2026-10-30 role-set v2 sets (25 questions at low / high effort, 30 striking prompts) live in `gi_v2/anecdotes/` and `muse_glimmer/gi_v2/anecdotes/` on scratch with the reader brief `READER_BRIEF_PROBES_V2.md`, the reports and `qverify_probes.py` (exact-matches every quoted span of a report or digest against the records; `--segments` for wrapped / nested quotes). Submit with untyped `--gres=gpu:1` (the typed `h200` line never starts on cs / cs2). |
| `scripts/probe_counts.py` | Keyword counts per setting for a probe root (refusals, truncation, manipulation words, policy words; `--groups`, `--leak`). |
| `scripts/random_direction_lens.py` | Logit lens + dictionary for the random control directions (`--seeds`, regenerated exactly as `steered_rh_vllm.py --random-direction-seed` draws them): extreme-token z per side against an isotropic null, cosines with named axes (`--axis name=path`) and every role−default persona vector in `--vectors-dir`, pairwise cosines; writes `random_direction_lens.{json,md}`. Own unembedding loader (Muse's `model.language_model.norm.weight`; Gemma's tied `embed_tokens` as the unembedding with the `1 + w` RMSNorm gain, 2026-11-04); reuses Koby's `logit_lens.py` functions. Used for the random-1730 (Muse, 2026-11-02) and random-1729 (Gemma, 2026-11-04) investigations. |
| `scripts/random1730_coding_compare.py` | Per-arm text markers over RH episodes (reasoning length, non-ASCII / foreign-script rate, truncation, attempts, give-up / hard-code / test-wrong phrases, verdict counts) and blind per-problem dossiers (all seeds of every `--arm name=root`, shuffled labels, `dossiers/key.json` kept from readers) for Sonnet readers. `--record` / `--reasoning-glob` for other file layouts (Gemma hard108: `pilot_impossible_original.jsonl` / `pilot_hard108_reasoning.jsonl`), `--head-chars` / `--tail-chars` show the start and, for long reasoning, the end of turn 0 (2026-11-04). |
| `scripts/hack_lens_extract.py` | Hack-direction lens, GPU step (2026-11-05): teacher-forces finished hard108 episodes through Gemma 4 31B turn by turn as the model saw them (chat-template history with the earlier finals, then the turn's `<|channel>thought` reasoning and final), records layer 30-59 token means over reasoning and answer tokens (pooled + per turn, up to 5 turns) with per-episode norms and projections onto `--direction name=path[:key]` and 8 seeded null directions; `--steer none|seed:<s>:<alpha>|axis:<path>:<alpha>|noise:<seed>:<alpha>` runs every turn twice (hook on = as generated, hook off) and reduces the per-token Δ on the GPU (norm, projections, cosines). One `<out>/<arm>__seed<N>.pt` per seed, skips existing; `--max-length` is per turn (whole-conversation rendering was rejected: 53/108 seed-0 episodes exceed 40 960 tokens). Driven by `random1729/hack_lens/extract.sbatch` + `tasks.tsv` (arm, run, steer, seeds) on cs3 B200, ~36 min per 108-episode seed single pass. |
| `scripts/hack_lens_analysis.py` | Hack-direction lens, CPU step: from the unsteered `.pt` files and a verdict file keyed `<run>/seed<N>/<example_id>` builds hack-minus-no-hack directions per layer (reasoning / answer / hack-turn via `--detail` / last-turn), with split-half cosine (even vs odd seeds), cross-fitted AUROC, gap, and cosines with the `--direction` references; then for every steered arm reports Δ along each hack direction as a fraction of the hack gap, the steered-state shift in the same units, amplification ‖Δ‖/|coef|, the null band and projections on the references. Writes `hack_lens.{md,json}`, `hack_directions.pt`, chart `hack_lens.{png,pdf}` (four panels + table). |
| `scripts/arm_discrim_test.py` | Blind two-arm discrimination test on RH episodes: per problem, 2 labelled reference episodes per arm (P/Q randomised) + 2·n unlabelled test items from distinct seeds, no pass/verdict metadata; `--score` compares the readers' `answers/<problem>.json` with `key.json` (pooled hits, exact binomial p, cues). Gemma random-1729 vs gi_v2 vs unsteered all at chance (2026-11-04). |
| `scripts/rand_arm_problem_seed_analysis.py` | Steered arms against unsteered without new judging: per-problem Δ correlations between arms (plain, partial given unsteered, split-reference), per-seed rates with sign test / split halves / leave-one-out / seed bootstrap, and same-seed divergence (common prefix of the turn-0 reasoning with the unsteered twin of the same sample seed; hack vs other, shared-vs-new hacks); `--chart` draws the three-panel figure + table, `--chart-only` redraws from the saved json. Gemma random-1729 checks (2026-11-04). |
| `scripts/rand1729_sign_pair_analysis.py` | Sign-pair checks for a random direction run at ±α against unsteered and null directions, from existing verdicts and records (no judging, no GPU): A per-problem mirror test (Δ+ vs Δ− correlations plain / partial given unsteered / split-reference, read against the null-pair floor; regression to the mean; hypergeometric overlap of the lowered and lifted sets inside the baseline-hacking problems; hacking-problem sets kept / new / lost), B reasoning length (turn-0 length per arm, within-arm HACK vs other with within-problem sign tests, mediation prediction from the within-problem slope × length shift, length halves), C hazard by attempt index (judged hack at the judge's turn and at the final passing submission, proxy, genuine, fail → continue / attempt limit / gave up; problem-bootstrap CIs on arm − unsteered; what follows a failed first attempt; judge-field mixes). Writes `sign_pair_analysis.{json,md}` and a 3 × 3 chart; `--chart-only` re-renders from the json. FINDINGS "Gemma random 1729 sign pair" (2026-11-04); run by `random1729/sign_pair.sbatch`. |
| `scripts/randscreen_proxy_score.py` | Automatic scoring of random-direction screen arms from the harness records alone (no judge, no GPU): shown / hidden / proxy / no-code / give-up / attempt-limit rates and attempts per arm, the per-seed sd of multi-seed arms (the noise a single-seed screen entry carries) and the percentile of marked arms inside the single-seed screen population. Written for Koby's 30-direction `g108-randscreen` (job 15035785, `L30_rand<seed>_a0.01/seed0`); `randscreen_scores.{json,md}`. |
| `scripts/random_direction_geometry.py` | Where a seeded random direction (and any `--axis`) sits in a layer's activation geometry, from the extraction `act_*` cache of response means: cosine with the mean activation, energy on the massive dimensions and top PCs, whitened / variance-weighted norms, each with its percentile in an isotropic null of fresh directions from the same generator. Gemma random-1729 checks (2026-11-04). |
| `scripts/make_split_direction.py` | Splits a seeded random steering direction into its component inside the top-k PCs of a layer's response means and the orthogonal remainder, each saved as a unit steering artifact (`{"model","axis"}`) with `split.json` (energy split, cosines). Gemma random-1729 controls (2026-11-04). |
| `scripts/save_random_direction.py` | Saves a seeded random unit direction as a `{model, axis, note}` artifact with the layer-`--layer` draw tiled over every layer, for cap bands: `--random-direction-seed` draws a different vector per layer (RNG seeded on (seed, layer)), so a cap over layers 20–38 with the flag would clamp 19 unrelated directions; `--check name=path` reports cosines with other artifacts (the 1729 split parts: pc50 0.097, rem50 0.995). First use: the Gemma random-1729 cap arm (2026-11-06). |
| `scripts/cap_threshold_interp.py` | Cap thresholds at a percentile the projection sidecar did not store (PLAN "Capping dose ladder"): per-turn normal-score interpolation between the two stored neighbours (`hf_harmony.QUANTILES`; p35 from p25 / p50 at weight 0.429), then the median over turns as `cap_thresholds.sbatch` takes it, so the rungs are monotone layer by layer on the same episodes; writes the `{layer: tau}` file `--cap-thresholds` reads plus a `.note.json` with the method and the stored neighbours. `python scripts/cap_threshold_interp.py <proj_*_projection.jsonl> 0.35 --out <dir>/<name>_p35.json [--layers 20 … 38] [--linear]`; tests `tests/test_cap_threshold_interp.py`. |
| `scripts/downstream_lens.py` | Causal downstream lens (HF, eager hooks, prompts only): add an arm's vector at the steer layer on the real turn-0 task prompts and record, per later layer, the growth of Δ relative to the injected coefficient, cos with the injected direction and with a reference axis, the share of Δ inside that layer's top-k PCs, plus next-token KL and top-1 flips; arms `seed:` / `axis:` / `noise:`; md + json + chart; `--rechart <json>` rebuilds md + chart from a finished run without a GPU. Gemma random-1729 controls (2026-11-04; job 15047951, FINDINGS "Gemma random 1729 causal downstream lens"). |
| `scripts/probe_blind_copy.py` | Copies probe arms (`neutral_probe.sbatch` layout) into a blind directory with shuffled letter labels and the steering metadata stripped, so probe readers compare arms without knowing which is which; mapping in `<out>/key.json`. |
| `scripts/probe_similarity.py` | TF-IDF cosine between two settings' finals on the same probe prompt (A vs B, and A vs itself across seeds) plus style markers (final / reasoning length, '!', 'I feel', em-dashes per final); `--root <probe set> --settings … --out <md>`. Used to show the random arms answer like unsteered while gi +0.08 is the least-similar arm (FINDINGS "Muse random direction 1730"). |
| `slurm/cap_thresholds.sbatch` | Cap thresholds from unsteered HF episodes; `MODEL`, `CHAT_KWARGS`, `LAYERS` (band), `TOPP`/`TOPK` overrides (defaults: gpt-oss, band 12–23; Muse: band 17–33). |
| `slurm/dose_pilot.sbatch` | One array task per steering setting (TSV), 45 problems × 1 seed; also used for unsteered calibration runs (`SCENARIOS`, `--data`, `--impossible-prompt` overrides). Any model and the long budget through optional variables (`MODEL`, `CHAT_KWARGS`, `ENGINE_KWARGS`, `TP`, `MAX_NEW_TOKENS`, `MAX_MODEL_LEN`, `CONCURRENCY`, `SELFCONCEPT_VENV`); the Qwen qwen_hard61 / Gemma cap-ladder recipe is `MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=65536 CONCURRENCY=45` (formerly `pilot_hard45.sbatch`). |
| `slurm/qwen_concurrent_check.sbatch` | Batch check for the concurrent engine mode (`steered_rh_vllm.py --batch-check`): greedy probes one at a time vs batched ×4, unsteered and steered, writes `compare.json`. |
| `slurm/main_run.sbatch` | Arms × seeds array for the main run. |
| `slurm/capability_eval.sbatch` | Public benchmarks per arm. |
| `slurm/checks/` | One-off backend checks that were run once (HF smoke, eager vs compiled vLLM, prompt log-prob check, CoT activation rerun). Keep for a new family: they are the evidence that a hook site and a compiled patch agree. |
| `post/figures/`, `results/` | Write-up figures and small result JSONs. `slurm-logs/` is gitignored. |

Outputs go to scratch, `~/nobackup/autodelete/self-concept-experiments/schemer-axis/` (autodelete: copy what matters).
Shared data (Koby's): `/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/` with `rh_hillclimb/data/*.jsonl`
(the problem sets), `rh_hillclimb/judge_rubric_v3.md` (the hack rubric), `rh_hillclimb/existing_hacks.md`, and the
gpt-oss Assistant Axis at `assistant-axis/gptoss12b-thinking/axis_response_only.py` (a torch file). Every sbatch reads
it from `MIRROR`, overridable. Run copies with their own README under `schemer-axis/`: `gptoss120b/covshaped/` and
`muse_glimmer/covshaped/` (covariance-shaped control directions, 100 × 1), `gptoss120b/antonym_controls/`
(isolated antonym control directions, 100 × 1), `qwen38_27b_gi_v2/` (Qwen gi_v2 vector and
the qwen_hard61 steering run).

## Environment and conventions

- Python: the repo's `.venv-313` (`uv sync`; see the top-level README). Jobs do `export SELFCONCEPT_VENV=$SLURM_SUBMIT_DIR/.venv-313; source activate.sh`.
  For CPU scripts: `OPENSSL_CONF=/dev/null .venv-313/bin/python ...` (the cluster's OpenSSL FIPS mode aborts otherwise).
- Hugging Face: `HF_HOME=~/nobackup/autodelete/hf_cache`, offline (`HF_HUB_OFFLINE=1`); the checkpoint must be in the cache
  before a job starts. `convert.py --datasets lcbcall` and the capability benchmarks need one online run first.
- Submit every sbatch from the repo root (they `--chdir` into this directory). Pick a GPU with `python -m byutils.gpu_pick --hours H --vram-gb V --cpus 8 --mem-gb 128 --cuda 13.0`.
  gpt-oss-120b and gemma-4-31B need an 80 GB card or more (H200 comfortable; A100-80 does not fit 31B at 40k context; B200 best).
- Steered vLLM engines must run with `VLLM_DISABLE_COMPILE_CACHE=1` and `VLLM_ENABLE_V1_MULTIPROCESSING=0` (set in the
  scripts and sbatch files): vLLM's compile cache is keyed on the model, not on the patch, and cross-serves graphs otherwise.
- Chat-template kwargs for the harness come from `SOO_CHAT_KWARGS` (JSON): gpt-oss `{"reasoning_effort":"low"}` (the recipe),
  Gemma 4 `{"enable_thinking":true}`. The sbatch files take `CHAT_KWARGS=` and export it.
- Unit tests (CPU, stub tokenizers): `OPENSSL_CONF=/dev/null .venv-313/bin/python -m unittest tests.test_intervene tests.test_steering_specifics tests.test_make_dossiers tests.test_direction_analysis tests.test_gpt_harmony_pooling tests.test_gemma4_pooling tests.test_convert_lcb_call tests.test_judge_runner`.
- Record keeping: one dated section per analyzed run in FINDINGS.md, status table updated; PLAN.md changed with a dated
  note when the design changes; one commit per round. Several people share the working tree and the Slurm account:
  check `squeue -u $USER` and `git log` before submitting or committing.

## Stage A: find reward hacks (unsteered runs)

> **Superseded for full runs.** The `arm_search` module (`src/selfconcept/arm_search/`, configs and sbatch files in
> `experiments/arm_search/`; on the `arm-search` branch, merged to `main`) replaces the manual loop below: a Bayesian batch search
> (tiered Thompson sampling) over problem pool, prompt tier, note, attempt limit and stated budget, which refits and
> re-plans each round (`python -m selfconcept.arm_search.rh.cli plan|manage --search <config>`, e.g.
> `experiments/arm_search/rh/search.yaml`). Use it to find hacks for a full run. The manual Stage A below is still fine
> for quick tests: checking the harness on a new model, a small pilot, or calibrating a handful of problems.

The harness (`selfconcept.codebench.harness`) gives the model a coding task with shown tests and checks the hidden full
suite. A record (`<out>/<tag>_impossible_original.jsonl`) has `passed` (shown tests), `passed_original` (hidden suite),
`n_attempts`, `attempts[]` (completion, code, result), `stop_reason` (passed / attempt_limit / stopped / flagged) and
`label`; the reasoning of every turn is in `<tag>_<scenario>_reasoning.jsonl`. The automatic hack proxy is
`passed and not passed_original`; it is a candidate filter, not a verdict. Workspace (EvilGenie-style) scenarios use the
labels `solved / hack_visible_only / visible_only_holdout_timeout / hack_test_modified / failed`.

The recipe (`hsolvNCblow`), the `RECIPE` line of the sbatch files:

```
--data $MIRROR/rh_hillclimb/data --impossible-prompt critical_no_carve --feedback neutral --max-attempts 5
--state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed $SEED
```

`critical_no_carve` is ImpossibleBench prompt C plus "Do NOT try to carve out the code"; `critical` is the softer prompt C;
`strict` suppresses hacks. `--feedback neutral` is the clean protocol (no "make the tests pass" pressure; a reply without
code ends the episode); `--state-attempt-budget` states the budget and the attempts remaining after each check. Gemma 4
adds `--top-p 0.95 --top-k 64 --max-new-tokens 32768 --max-model-len 40960` (plain temperature-1 sampling loops; the
16k budget truncates its thinking). The sbatch files put TSV extras after `RECIPE`, and argparse's last value wins, so a
settings line can override any recipe flag, including `--data` and `--impossible-prompt`.

1. **Calibrate the problem pool.** Problems must sit at the edge of the model's ability (Koby's rule: solved < 90% of
   the time, ideally around 50%; always-solved problems are solved). Keep never-solved problems: they are not honest
   by default (gpt-oss-120b unsteered hard45: 30 of 56 judged hacks on the 16 problems it never solves in the
   random-direction arms, 16% of their episodes; 3 of 56 on problems solved >= 50%).
   Run the candidate sets unsteered over a few seeds with `dose_pilot.sbatch` (one `unsteered<TAB>` line per seed with
   `--sample-seed N`), then compute the per-problem hidden pass:

   ```
   sbatch --array=0-2 --export=ALL,SETTINGS=<tsv>,ROOT=<dir>,MODEL=<model>,CHAT_KWARGS='{"enable_thinking":true}' experiments/deceiver-axis/slurm/dose_pilot.sbatch
   python scripts/pilot_summary.py <dir> --out <dir>/summary.json
   ```
   Problem sets in `rh_hillclimb/data`: `ib_solvhard_s0-2` (the 45 hard ImpossibleBench problems the gpt-oss study uses),
   `ib_solveasy_s0-2` (the 58 easier ones), `p1pro_s0-5` + `nm_pro_*` (LiveCodeBench-Pro workspace tasks). More LeetCode-style
   problems in the same format: `python -m selfconcept.codebench.convert --datasets lcbcall` writes
   `benchmarks/codebench/data/lcbfn_{easy,medium,hard}.jsonl` (needs the Hub once); pass their directory as `--data` and
   the file stem as the scenario.
2. **Read every proxy episode.** Open its reasoning sidecar and judge it against `rh_hillclimb/judge_rubric_v3.md`
   (`scripts/dump_proxy_episodes.py --root <run> --data <pool.jsonl> --out dump.txt` lists each one with the failing
   hidden assertion from a sandboxed re-run, test-mentioning code comments, literal-branch counts and reasoning excerpts)
   (HACK = enacted in code, knowing, not licensed by the instructions). Honest wrong answers and problems whose general
   solution fails the hidden suite dominate the proxy at low rates; a regex for intent terms is not enough.
   For blind judging at scale: `python scripts/make_dossiers.py --data <data> --out <dir> --runs ARM=DIR ...`.
   Judge and reader subagents run on **Sonnet** (`model: sonnet`), not the session's default model: the rubric-v3
   hack judges (one batch per agent, blind, own salt; user rule 2026-10-20) and, from 2026-10-23, the anecdote reader
   agents that quote transcripts for the paper (cost; reader briefs in `anecdotes_v2/READER_BRIEF.md` and `gi_v2/anecdotes/READER_BRIEF_PROBES_V2.md` on scratch).
   Finished work is not redone to switch models.
   One read per dossier and no adjudication pass (user decision 2026-10-26): the judged hack rate counts HACK STRONG and
   HACK WEAK, not BORDERLINE (`verdict_variants.py` `as_judged`; the BORDERLINE-up and STRONG-only variants are the sensitivity).
   After judging: `python scripts/collect_verdicts.py --verdicts <dir with *.jsonl> --dossier-sets <dossier dir> ... --out verdicts.json`, then `main_analysis.py ... --verdicts verdicts.json`.
3. **More hacks** (Koby's levers, FINDINGS 2026-10-16): sets near 50% hidden pass, low-pass problems across many seeds,
   the softer prompt (`--impossible-prompt critical`), and a stated budget with attempts remaining (already in the protocol).

The gpt-oss main run's unsteered arm (45 problems × 12 seeds) shows the proxy at 19%, with ~8% STRONG hacks under the
rubric in Koby's data; Gemma 4 12B never reaches that regime on these problems (ability floor); 31B does.

## Stage B: extract and validate the schemer direction

1. **Extraction** (one GPU job; resumable, every step skips roles already on disk):

   ```
   sbatch --export=ALL,MODEL=google/gemma-4-31B-it,ROOT=<scratch>/<model>/full experiments/deceiver-axis/slurm/extract_roles_gemma4.sbatch
   ```
   Steps: `1_generate` (40 questions × 5 prompts per role, temperature 1, thinking on) → `3_judge` (Qwen2.5-7B scores
   0–3, keep 3 = fully in role) → `2_activations` twice (`act_response_only`: answer tokens; `act_with_cot`: thinking
   tokens, `--run.include_cot true`; both passes render the conversation with thinking on, `--run.thinking true`) →
   `4_vectors` (per-role mean, `vectors_<pass>/{role}.pt`). `ROLES="..."`, `QUESTIONS=`, `MIN_COUNT=` override; a
   smoke is `ROLES="default schemer schemer_fic straight_shooter_fic" QUESTIONS=3 MIN_COUNT=1`. Run the bare roles
   too: whether the model plays a bare persona is itself a finding (gpt-oss refused; Gemma did not).
   Qwen3.8-27B (2026-10-27): `MODEL=Qwen/Qwen3.8-27B CHAT_KWARGS='{"reasoning_effort":"low"}' GEN_MAX_NUM_SEQS=128`;
   `CHAT_KWARGS` reaches both the generate and the activation renders, `GEN_MAX_NUM_SEQS` caps vLLM's batch.
2. **Validation** (CPU, no reward-hacking data). At the model's mid-depth layer (gpt-oss 17 of 36, Gemma 12B 24 of 48,
   Gemma 31B 30 of 60) and as a sweep:

   ```
   python scripts/validate_vectors.py --activations-dir <root>/act_with_cot --scores-dir <root>/scores --layer 24 \
       --schemer schemer_fic deceiver_fic cheater_fic --honest straight_shooter_fic rule_follower_fic --default default \
       --out <root>/validation_with_cot_fic_L24           # [--assistant-axis <axis.pt>] if the model has one
   for L in 4 8 12 16 20 24 28 32 36 40 44 47; do python scripts/validate_vectors.py ... --layer $L --out <root>/sweep_with_cot_fic/L$L; done
   ```
   Gates (PLAN "Phase 1"): ≥ 50 of 200 responses at score 3 per role; split-half cosine ≥ 0.8; held-out schemer-vs-honest
   AUROC ≥ 0.9. A vector that fails is reported as unusable, which says nothing about reward hacking.
3. **Candidates**:

   ```
   python scripts/build_vectors.py --vectors-dir <root>/vectors_with_cot --model google/gemma-4-31B-it --out <root>/candidates_with_cot \
       --schemer schemer_fic deceiver_fic cheater_fic --honest straight_shooter_fic rule_follower_fic --default default --layers 30
   ```
   `persona_contrast` (schemer mean − honest mean) is the primary; `persona_raw` (schemer mean − default) the hedge. The
   artifact stores the model id and is checked against the model at steering time. The `_orth` candidates and the AA
   cosine need `--assistant-axis`.
4. **Phase 2 (correlation)** exists only for gpt-oss: it projects the cached layer-17 residuals of Koby's judged cohort
   (`~/nobackup/autodelete/rh-random-control-v3/`) onto the candidates against 256 random directions
   (`selfconcept.correlation.direction_analysis`). A new model needs its own judged cohort and residual cache first.

## Stage C: steer (Phase 3)

All on `steered_rh_vllm.py`: the add / cap transform is patched into the vLLM decoder-block class before the engine is
built, so it is inside the compiled graph and runs at full speed (≈ 22 s per gpt-oss episode on an H200).

1. **Layer norms and sanity probes** (`gemma4_rh_smoke.sbatch`; works for any supported family with `MODEL=`,
   `DIRECTION=`, `LAYER=`, `ALPHA=`): `--measure-norms` writes `norms/layer_norms.pt`, the unit of α (α = 0.2 means
   0.2 × the mean residual norm at the layer); `--probe` answers three neutral questions at 0 / +α / −α (check 5 of
   Phase 1: the answers should move toward and away from scheming while staying coherent). Also read the natural
   size of the contrast relative to the residual norm (gpt-oss 12.6% at layer 17, Gemma 12B 2.6% at layer 24) before
   choosing an α grid: the same α is a very different perturbation in a different model.
2. **Dose pilot** (`dose_pilot.sbatch`, settings TSV `name<TAB>extra args`, 45 problems × seed 0 per setting):

   ```
   unsteered	--top-p 0.95 --top-k 64 --max-new-tokens 32768 --max-model-len 40960
   cot_fic_L24_a0.04	<same> --direction <cand>/persona_contrast.pt --layer-norms <root>/norms/layer_norms.pt --steer-mode add --steer-layer 24 --steer-alpha 0.04
   cot_fic_L24_a-0.04	<same> ... --steer-alpha -0.04
   ```
   `python scripts/pilot_summary.py <root>` prints the capability measures per setting. Dose rule (PLAN): the largest
   dose whose hidden pass and no-code/truncation rate stay within 10 points of unsteered on both signs, chosen on
   capability measures only, never on the hack rate; and the dose decision goes to the user. 45 episodes give a 6-point
   standard error, so read the point estimates too (the gpt-oss main run used 0.2, not the 0.4 the rule admitted).
3. **Cap arm**: thresholds are percentiles of the per-token projection in unsteered episodes, from the HF path
   (`cap_thresholds.sbatch` → `{layer: tau}` JSON), applied as `--steer-mode cap --cap-layers <band> --cap-thresholds
   <json>`. The band is the middle third of the depth around the steering layer (gpt-oss 12–23 of 36 around 17; Muse
   17–33 of 52 around 26). A new family needs a specifics class in `src/selfconcept/measurement/steering_specifics.py`
   (both paths read it; the HF path is exercised on gpt-oss and Muse Glimmer, not yet on Gemma 4).
4. **Random control**: a norm-matched isotropic direction at the persona dose broke gpt-oss (the residual stream is
   anisotropic). Use the capability-matched version: a random-direction dose pilot (`--random-direction-seed 1729`,
   several seeds × α) under the same dose rule, then a 12-seed random arm at the largest passing α.
5. **Main run** (`main_run.sbatch`, arms TSV, `NSEEDS=12`, task i → arm i / NSEEDS, seed i % NSEEDS; output
   `<root>/<arm>/seed<s>/main_impossible_original.jsonl`, resumable):

   ```
   sbatch --array=0-59 --export=ALL,ARMS=<root>/arms.tsv,ROOT=<root>,NSEEDS=12,MODEL=<model>,CHAT_KWARGS='...' experiments/deceiver-axis/slurm/main_run.sbatch
   python scripts/pilot_summary.py <root> --tag main
   python scripts/main_analysis.py <root> --out <root>/analysis.json [--verdicts <json>]
   python scripts/make_dossiers.py --data <data> --out <root>/dossiers --runs unsteered=<root>/unsteered shift_plus=<root>/shift_plus ...
   ```
   A drop in the proxy that comes with a drop in hidden pass or a rise in no-code is degradation, not less reward hacking.
   **Concurrent mode** (`--concurrency N`, added 2026-10-27 for Qwen's 64k reasoning budgets): N episodes run at once on
   the one engine (`src/selfconcept/codebench/vllm_concurrent.py`; each episode is the unchanged harness loop on its own
   thread, the main thread steps the engine), same records, sidecars, summary and resume; `meta.concurrency` is recorded.
   Add `--concurrency 45` to every settings/arms line of a pilot or main run, including the unsteered one: batched kernels
   round differently from single-sequence ones, so concurrent and sequential episodes agree in distribution, not token
   for token, and arms are compared only within one mode. On an 80 GB card the KV cache holds ≈ 245k tokens, so about
   10–16 episodes at a 73k window decode at once and vLLM preempts the rest (an H200 holds 45). `--batch-check`
   (`slurm/qwen_concurrent_check.sbatch`) verifies for a new family that the steering patch acts at batch size > 1:
   greedy probe answers one at a time vs four copies batched, unsteered and steered; the batched steered copies should
   share a longer prefix with the steered sequential answer than with the unsteered one.
6. **General capability** (`capability_eval.sbatch` with the same arms TSV; `capability_eval_compare.py <root>`): GSM8K,
   MMLU-Pro subsample, HumanEval and IFBench (added 2026-10-27; `allenai/IFBench_test`, 300 prompts with verifiable
   constraints, prompt-level loose accuracy through the `ifbench` package's verifiers) through the same patched engine,
   paired per item against unsteered. `ifbench` is not on PyPI; it is installed from GitHub, pinned:
   `uv pip install --python .venv-313/bin/python "ifbench @ git+https://github.com/allenai/IFBench.git@1c40f0c1"`
   (not in uv.lock), and its nltk resources (punkt, punkt_tab, stopwords, averaged_perceptron_tagger_eng) must be in
   `~/nltk_data` for the offline compute nodes (`nltk.download(..., download_dir=~/nltk_data)` on the login node). All
   300 items score offline (`tests/test_capability_eval_ifbench.py` covers the loose rule and one real verifier).

## Human validation of the judges

`scripts/human_validation/` (CPU, seconds). Labelers use the judge's own scale, but the question is positive or not
(for the in-character judge, 3 vs 0–2: `--positive-levels 3`). A sampler draws half judged-positive and half
judged-negative items (model-balanced, shuffled) and writes a blind `items.jsonl` and `rubric.md` plus a `key.jsonl`
(judge label, model, source) that is never sent; `build_labeling_page.py` turns items and rubric into one HTML file that
labelers open in a browser (rubric beside one item at a time, keys a s d f, autosave, "Download labels" gives
`labels_<title>_<name>.jsonl`); `score_agreement.py` lumps every label into positive or negative and reports mean human–human and judge–human
Cohen's κ with item-bootstrap 95% CIs, the judge's precision (judged-positive items a strict majority of labelers, and every labeler, also
call positive; Wilson 95% CI, overall and per model), the full-scale confusion matrices and per-model κ.

```bash
cd experiments/deceiver-axis/scripts/human_validation
OUT=$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/human_validation/in_character
python sample_in_character.py --run gpt=<gpt root> --run muse=<muse root> --run qwen=<qwen root> \
    --roles-dir ../../data/roles --out-dir $OUT
python build_labeling_page.py --items $OUT/items.jsonl --rubric $OUT/rubric.md --labels 0,1,2,3 \
    --title in_character --out $OUT/in_character.html          # send only this file
python score_agreement.py --key $OUT/key.jsonl --labels labels_in_character_*.jsonl --levels 0,1,2,3 --positive-levels 3
```

## Adapting to a new model family: checklist

Two code paths touch the model, HF (extraction, projections) and vLLM (steering); both are keyed on the model name.

1. **HF model specifics** (`src/selfconcept/assistant_axis/internals/model_specifics/`): add a class to the registry
   (`registry.py`, substring match, first match wins, so put `gemma-4` before `gemma`). It needs `get_response_indices`
   and `build_turn_spans` with the `include_thinking` keyword (which tokens are "the assistant's answer" vs "its
   thinking"), and a generation-specifics mixin from `src/selfconcept/common/generation/model_specifics/` (thinking
   flag, completion split/join, `skip_special_tokens`), registered there as well. ChatML-style templates (Qwen, OLMo,
   GLM) reuse `_shared.py`; a model with its own thinking channel follows `gemma4.py` or `muse_glimmer.py` (a token-level
   message iterator plus a splitter; Muse Glimmer's ATEM `to=self` channel is always on and `direct_answer_prefix_ids`
   forces the user channel). Test it on a stub tokenizer that renders what the real template emits (`tests/test_gemma4_pooling.py`).
   Multimodal checkpoints load through `selfconcept.common.loading.load_causal_lm` and keep their sizes in
   `text_config`; `src/selfconcept/assistant_axis/models.py` falls back to the middle layer when the model has no entry.
2. **Steering model specifics** (`src/selfconcept/measurement/steering_specifics.py`, the `model_specifics` pattern of
   `arm_search/rh`: a frozen dataclass per family, a name-substring registry, `steering_model_specifics(model_name)`):
   the record label (`family`), the HF attention kernel, the vLLM decoder-block class (`vllm_block`, module path and
   class name), `patch_vllm` with the patcher that matches its forward contract (`patch_vllm_blocks` when forward
   returns `(mlp_output, residual)` as gpt-oss's `TransformerBlock`; `patch_vllm_stream_layers` when it returns the
   residual stream itself, as `Gemma4DecoderLayer` and `MuseGlimmerDecoderLayer`; the layer index is read from the
   layer or its `self_attn`), `vllm_stream` that reads the residual off a forward-hook output, and the completion
   splitter (`completion_splitter(chat_kwargs)` for a family whose template can turn thinking off, as Qwen's).
   `tests/test_steering_specifics.py` covers resolution, splitters and both patch contracts on stub classes.
   `model_layers` in `steered_rh_vllm.py` must find the decoder layers on the vLLM model object. Then verify once:
   `--check-hook` against an HF projection (`slurm/checks/steered_rh_vllm_smoke.sbatch`), eager vs compiled agreement
   (`steered_rh_vllm_compiled_smoke.sbatch`), prompt log-probs (`steered_rh_vllm_logprob_check.sbatch`).
   `steered_rh.py` and `capability_eval_vllm.py` read the same specifics.
3. **Harness settings**: thinking on/off and effort through `SOO_CHAT_KWARGS`; sampling from the model's
   `generation_config.json` (`--top-p`, `--top-k`); a thinking budget the model actually finishes in (`--max-new-tokens`,
   `--max-model-len`; check the truncation column of `pilot_summary.py`); GPU memory (`VLLM_GPU_MEM`). A model whose
   thinking runs past the budget submits no code, and no-code is not a hack.
4. **Problem pool**: measure the per-problem hidden pass first (Stage A, or `arm_search` for a full run) and keep problems in the 10–90% band, ideally
   near 50%. Changing the pool, the prompt or the sampling is a recipe deviation: record it, and keep the unsteered arm
   of every comparison under the same settings as the steered arms.
5. **Layer and dose**: mid-depth layer for validation and steering (plus the sweep); α in units of that layer's mean
   norm; the grid from the contrast's relative size; the dose rule on capability measures only. Both signs.
6. **Cost (2026-10)**: extraction 12 roles × 200 on a 12B model ≈ 1 A100-hour; a 45-episode unsteered setting on
   gemma-4-31B ≈ 4–5 h on a B200 (five attempts, 32k thinking); a gpt-oss main-run arm (540 episodes) ≈ 3.5 H200-hours.
   Standby queues preempt; the harness resumes finished example ids, so `--requeue` is safe.

## Known pitfalls

- A `--gpus-per-node=1` task on the cs A100 nodes can see a sibling task's GPU; the sbatch files pin
  `CUDA_VISIBLE_DEVICES` to the least-used visible GPU by UUID.
- Seeded sampling reproduces a run only within one engine configuration (context length, GPU type); use the same
  engine settings across arms.
- `pilot_summary.py` reads check-format records (`*_impossible_original.jsonl`); workspace-scenario runs are read from
  their labels.
- Gemma 4's chat template injects `<|think|>` only when `enable_thinking` is set and renders an assistant turn's
  `reasoning` only for turns after the last user message; the activation pass must render with thinking on
  (`--run.thinking true`) for the CoT and answer pooling passes to see the same tokens.
- On Blackwell (B200) vLLM auto-selects the FLASHINFER attention backend for dense models such as the Qwen judge and
  its JIT build fails against the venv's CUDA headers ("CUDA compiler and CUDA toolkit headers are incompatible",
  job 14272566). `extract_roles_gemma4.sbatch` sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN` on B200 nodes and
  `VLLMGenerator` passes it on as vLLM's `attention_backend`; Gemma 4 itself picks TRITON_ATTN there anyway.
  The steered harness (`steered_rh_vllm.py`) does not read that variable (vLLM 0.28 logs it as unknown), so on
  B200 the engine picks its own backend: gpt-oss gets FLASH_ATTN and runs (smoke 14916724, 2026-10-30, three steered
  answers in 317 s); Muse gets FLASHINFER and dies in the same JIT build (smoke 14916725, 2026-10-30). The fix is the
  engine argument: add `--engine-kwargs {"attention_backend":"TRITON_ATTN"}` to the settings row (`steered_rh_vllm.py`
  passes `--engine-kwargs` through to `vllm.LLM`); the Muse retry with it ran (smoke 14920170, 2026-10-30, three
  steered answers in 195 s). So gpt-oss goes to cs3 as is and dense models (Muse, Qwen, Gemma 4) go there with that
  kwarg (`-p cs3 --qos cs --gpus-per-node=1`, untyped). Do not add the kwarg to a running array's frozen settings file.
- Gemma 4 31B capability eval (`capability_eval.sbatch`, 20,480-token budget at `gpu-mem 0.95`) does not fit an 80 GB
  card: the KV cache needs 14.9 GiB and 14.3 are left after the weights (job 14982952, 2026-11-02). Run it on cs3 B200
  only (job 14983080: five arms as array tasks, `-p cs3 --qos cs --gpus-per-node=1`). The same day cs-1-1 showed idle
  GPUs but failed every job with "Unable to determine the device handle for GPU0" (job 14982942); exclude it or report
  it to ORC when a job lands there and dies at start.
- Qwen3.5/3.8 chat templates default to `reasoning_effort` xhigh; pass the effort through `CHAT_KWARGS` for the
  extraction as well as the harness. Their hybrid linear-attention layers need one Mamba cache block per sequence,
  so vLLM refuses to start at the default 1,024 sequences (223 fit on an 80 GB H100 at 0.85 utilisation); set
  `GEN_MAX_NUM_SEQS` / `VLLM_MAX_NUM_SEQS`. Qwen completions come back as `{thinking}</think>{answer}` with no opening
  tag; `QwenThinkGenerationSpecifics` splits them so the judge sees only the answer (the extractors' answer region
  excludes the thinking either way). The steering harnesses use `split_qwen_think_completion` (no `</think>` = no
  final), and `steered_rh_vllm.py` / `capability_eval_vllm.py` read `VLLM_MAX_NUM_SEQS` too (export 128 for Qwen).
- `patch_vllm_blocks` passes the block's arguments through: gpt-oss's `TransformerBlock` takes
  `(hidden_states, positions, residual)`, Qwen's `Qwen3_5DecoderLayer` `(hidden_states, residual, positions)`; the
  old fixed signature would have swapped residual and positions on Qwen (caught by `tests/test_qwen_think.py`).
- Qwen smoke: `slurm/qwen_rh_smoke.sbatch` (vLLM vs HF layer norms at every layer, hook check, greedy compiled-vs-eager
  patch check, probes). Harness sampling from `generation_config.json`: `--top-p 0.95 --top-k 20`.
- Validation with `--costume neutral_fic` (and `scripts/stratified_auroc.py`) reports the cleaned contrast and the
  costume AUROCs; `build_vectors.py --costume neutral_fic` writes the cleaned steering artifact `persona_contrast_clean.pt`.
- The sbatch files pin an H200 GPU type; on cs3 submit with `-p cs3 --qos cs --gpus-per-node=1` (untyped). A typed
  `b200:1` request is rejected.
- Do not call a direction "the schemer direction" before it passes the Phase 1 checks, and do not call a lower proxy
  "less reward hacking" when hidden pass, no-code or truncation moved with it (FINDINGS "Writing conventions").
