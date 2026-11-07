# Port review queue (branch deceiver-axis-port)

Plan: `~/nobackup/autodelete/self-concept-experiments/schemer-axis/notes/merge_survey_2026-11-06.md`. Each row is one
commit; tick or comment, follow-ups land as new commits (no rebase/amend). `port:` rows are mechanical moves, skim only.
Nothing is pushed or merged until every row is ticked.

## Checkpoint 1: schemer-pilot-mistral-nemotron + uncommitted changes

| Done | Commit | Kind | What | Alternative | Look at | Verified |
|---|---|---|---|---|---|---|
| [ ] | 1ca298a | port | Pi harness (pi_agent, submission_budget.ts, pilot_pi.sbatch), LCB-Pro / EvilGenie check converters, `llm.get_tokenizer()` in vllm_harmony | — | src/selfconcept/codebench/, slurm/pilot_pi.sbatch | applied cleanly; pyright: one new error, fixed in 82ecc93 |
| [ ] | 822330b | port | README calibration-rule fix (0903f53) | — | README.md | applied cleanly |
| [ ] | 8248b86 | small | cap_thresholds.sbatch: SELFCONCEPT_VENV overridable | — | slurm/cap_thresholds.sbatch | default unchanged |
| [ ] | 5acada5 | keep | Gemma 4 readout spans (was uncommitted in the gemma-goodfire worktree) | drop | internals/model_specifics/gemma4.py, internals/readouts.py | pyright: no new errors |
| [ ] | 82ecc93 | fix | `lcb_call_case` accepts a WorkspaceTestCase | cast at the call site | codebench/convert.py:400 | pyright error gone |
| [ ] | 4c67e70 | decision | TP steering reads steering_specifics for any family; steered TP still gated to {gemma4} | worker stays Gemma-only | scripts/steering_worker.py, scripts/steered_rh_vllm.py (TP_STEERING_VERIFIED_FAMILIES, main's TP branch) | pyright no new errors; CLI gate on CPU. **Not yet run on GPUs after the refactor** |
| [ ] | 179f539 | decision | pilot_hard45 folded into Jansen's dose_pilot.sbatch (optional ENGINE_KWARGS, TP, MAX_NEW_TOKENS, MAX_MODEL_LEN, CONCURRENCY; PYTHONPATH = this checkout's src) | rename to concurrent_pilot.sbatch | slurm/dose_pilot.sbatch | bash -n; defaults leave Jansen's command line unchanged |
| [ ] | ab67d40 | review fix | Your 1ca298a / 82ecc93 comments: TypedDicts for Pi examples, records, recipe fields and budget summary; CheckExampleWithCounts for the converters; lcb_call_case takes WorkspaceTestCase only | — | codebench/pi_agent.py (types near the top, run_one), codebench/convert.py (CheckExampleWithCounts, lcb_call_case, decode_private_tests) | pyright: only pre-existing errors left; codebench unit tests pass |

Dropped (FINDINGS gets a line in checkpoint 3): Mistral Small 4 / Nemotron 3.5 / OLMo 3 Think / Step 3.7 Flash
families, pixtral_compat shim, Mistral sbatch notes. Discarded uncommitted changes: main steered_rh.py Gemma entry,
rand1729 random_direction_lens.py fix, rand1729 loglik / episode-PCA scripts (they stay in scratch).

## Checkpoint 2: analysis ports

| Done | Commit | Kind | What | Alternative | Look at | Verified |
|---|---|---|---|---|---|---|
| [ ] | f66d160 | decision | Hard-coding detector as a library module `selfconcept.codebench.hardcoded_cases`; `hardcode_scan.py` is a per-arm CLI for any model | keep the detector inside the script | src/selfconcept/codebench/hardcoded_cases.py, scripts/hardcode_scan.py | 100 gpt-oss covshaped arms: same episodes / proxy / hits / hidden pass as scratch |
| [ ] | bd4d1fe | port | `controls/control_percentile.py`: gi_v2's percentile among controls (rates, Fisher, dispersion, Spearman, shape-free and beta q intervals), any model; hack predicates + loaders move into hardcoded_cases | — | scripts/controls/control_percentile.py | Muse covshaped 100x1: every field equal to scratch analysis_100x1.json |
| [ ] | 53bad43 | decision | `scripts/controls/` + `data/controls/`: isolated_directions (CLI), controls_common (downstream read as a one-method protocol, gpt-oss + Muse), sample_pairs, check_sentences, subagent instructions, pairs and the 100 sentence files | matched alpha gpt-oss-only, no protocol | scripts/controls/controls_common.py (DownstreamReadSpecifics), scripts/controls/isolated_directions.py | pairs.jsonl reproduced; sentences valid; matched alpha 0.1904713 = scratch. GPU rerun 15131997: all 109 directions bit-identical to 15126400, summary identical (minus the dropped covshaped set) |
| [ ] | cfaff4d | decision | Honest hidden pass = main capability measure: new `honest_hidden_pass` outcome in main_analysis (ratio of stratified means, paired bootstrap, swap permutation), pilot_summary column, dose rule uses it | keep hidden pass in the dose rule; average per-problem conditional rates | scripts/main_analysis.py (analyse_honest_hidden_pass), scripts/pilot_summary.py (within_dose_rule) | gpt-oss ladder: rates and differences equal cap_vs_steer.json |
| [ ] | e45313e | port | cap_vs_steer.py + chart; the four models' runs in data/cap_vs_steer_setups.json | — | scripts/cap_vs_steer.py, data/cap_vs_steer_setups.json | JSON and PNG byte-identical to scratch |
| [ ] | 9babefe | port | qwen_hard61_build.py | — | scripts/qwen_hard61_build.py | five outputs byte-identical |
| [ ] | 441d5c5 | small | nltk as a direct dependency | — | pyproject.toml | lock adds only the two dependency lines (3.10.3 was already locked) |
| [ ] | 1988c58 | review fix | control_percentile: target-vs-population naming (--target, --target-direction, keys target / at_or_above_target); constants commented | — | scripts/controls/control_percentile.py (constants, summarize_measure) | Muse 100x1 unchanged apart from the renamed keys |
| [ ] | 1a55436 | review fix | One data/controls/sentences.json instead of 100 files; check_sentences combines the subagents' files; AISI constant notes control_spacing | — | data/controls/sentences.json, scripts/controls/check_sentences.py | rebuilt byte for byte; loader yields the same pairs |
| [ ] | c37b5b0 | review fix | Typed honest-hidden-pass helpers; overall hidden pass dropped from main_analysis defaults and pilot_summary | — | scripts/main_analysis.py (types above honest_shares), scripts/pilot_summary.py | output identical; note gi_main_chart / gemma_hard108_chart read hidden_pass, so pass --outcomes for them |
| [ ] | 0364d3b | review fix | cap_vs_steer chart writes the PNG only | — | scripts/cap_vs_steer_chart.py | PNG identical |

Noticed while porting, not changed: control_percentile's beta fit only allows mean control rates from 0.08 to 0.40
(the grid the gpt-oss simulations used). Muse's pooled proxy rate is 0.37, near that ceiling, so the beta interval on
that measure may be squeezed. Widening the grid would change the reported Muse numbers; your call.
Not ported: the covshaped tooling, the failed DoM builders, Gemma hard108 summarize / tabulate, Qwen pilot_scan.

## Checkpoint 3: docs
PLAN pre-registration + 2026-11-06 / 2026-11-07 amendments, FINDINGS entries, README share rows.
