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

## Checkpoint 2: analysis ports (next)
hardcode_scan, controls/ (isolated_directions, controls_common, control_percentile, sample_pairs, sentences),
honest hidden pass in main_analysis / pilot_summary, cap_vs_steer, qwen_hard61_build; numbers check against the Muse
100x1 and cap_vs_steer outputs.

## Checkpoint 3: docs
PLAN pre-registration + 2026-11-06 / 2026-11-07 amendments, FINDINGS entries, README share rows.
