# SOO scripts: current work, reusable recipes, and cleanup

Reviewed 2026-09-29 against [PLAN2.md](PLAN2.md), the round plans, Python imports,
snapshot-copy dependencies, and executable references throughout the repository.
The review covered 30 Python scripts and 165 Slurm wrappers. The scheduler showed
no jobs for the current user at the time of the review; that observation alone
does not establish that any experiment completed successfully.

## Retention rule

Keep scripts that could plausibly support future training or study 2. A completed
round, a hard-coded model, an old environment path, or lack of current callers is
not sufficient reason to delete a recipe. All 30 Python scripts are retained.
All training, generation, steering, capability, and representation-measurement
Slurm scripts are retained. Only ten duplicate judge wrappers were removed,
after their complete judge command sequences were reproduced by retained wrappers.

## Python tools kept

| Scripts in `scripts/` | Why they stay |
|---|---|
| `launch_steer_round.py`, `launch_bisect_round.py` | Most recent study-2 rounds; steering imports the bisection launcher's preflight helpers. Bisection supports both rounds. |
| `launch_adapter_round.py`, `launch_adapter_models.py` | Repeat adapter-collapse measurements, including future adapters and cross-model comparisons. Completed rounds are still useful controls. |
| `launch_constant_round.py`, `launch_layer_round.py` | Reusable constant-replacement and layer-localization interventions for future adapters. |
| `analyze_adapter_round.py`, `analyze_constant_round.py`, `analyze_layer_round.py`, `analyze_bisect_round.py`, `analyze_steer_round.py` | Reproduce saved round results. Bisection and steering analyses import the layer analysis; deleting it would break current work. |
| `make_constants.py`, `extract_band_deltas.py` | Used by current launchers and tests; band extraction imports the constants encoder. |
| `extract_steering.py`, `prepare_subspace_pilot.py` | Rebuild vectors and the study-2 pair dataset. |
| `layer_gap.py`, `scale_check.py` | Potential diagnostics for new training runs. The former's full-tensor MSE includes padding; the latter has model/cache-specific settings. Use `selfconcept.soo.measure_overlap` for study-2 held-out gap measurements. |
| `judge_apollo.py`, `judge_insider.py`, `reparse_sandbagging.py` | Shared outcome tools, also used by the CoT-ness/assistant-axis pipeline. |
| `parse_sandbagging.py`, `audit_sanity.py` | The audit extracts the legacy parser's functions to reproduce its missing-answer bug. Use the repaired parser for new sandbagging analysis. |
| `paired_tests.py`, `intent_language_check.py`, `rescore.py` | Matched comparisons, response interpretation, and reclassification of saved outputs. |
| `make_validation_sample.py`, `make_apollo_validation_sample.py` | Human/judge validation and independent-judge checks. |
| `apollo_convert.py`, `caps_steered.py`, `setup_scratch.py` | Dataset generation, capability regression checks, and artifact-directory setup. |

## Slurm recipes kept

There are now 155 `.sh` files. Counting each under its primary purpose:

| Purpose | Files | Reuse |
|---|---:|---|
| Training, often followed by evaluation | 28 | Model configs, seed/layer/loss sweeps, smoke tests, and original/agentic/mixed adapter recipes. |
| Behavior and steering evaluation | 88 | Baselines, mirrored tasks, dose/direction/random controls, temperature tests, and coding outcomes. |
| Judging | 20 | Parameterized batches, independent judge families, Kimi delimiters, and distinct coding/call-out protocols. |
| Extraction and representation measurement | 11 | Steering vectors, latent/gap measurements, and the frozen subspace pilot. |
| Capability evaluation | 8 | Regression checks for future trained or steered models. |

These are retained recipes, not a claim that every historical wrapper is ready
to launch unchanged. Older wrappers select `.venv`, whose interpreter in this
checkout is Python 3.11, while current source requires Python 3.13+. Many also
contain the old GPU re-selection preamble. The recent bisection/steering launchers
use `.venv-313`, freeze code and inputs, preserve the assigned GPU, and configure
the model cache. Adapt an older recipe's runtime setup before reusing it; the
cleanup does not alter environments or submit jobs.

## Consolidated judge wrappers

The four retained wrappers below now accept **explicit response-file paths** as
positional arguments. Paths are relative to `experiments/soo`, the job's working
directory. With no arguments, each keeps its original file selection. Argument
arrays preserve filenames with spaces; quoted wildcards are not expanded.
`code-judge-qwen38-val.sh` also accepts `OUT_DIR` to separate validation outputs.

| Removed wrapper | Retained replacement | Selection to reproduce the old job |
|---|---|---|
| `apollo-qwen38-judge.sh` | `apollo-judge.sh` | Qwen3.8 roleplaying response files. |
| `apollo-round2-judge.sh` | `apollo-judge.sh` | The eight round-2 Gemma roleplaying files. |
| `apollo-r3-judge-a.sh` | `apollo-judge.sh` | The sixteen Mistral/OLMo/Gemma-2/Muse roleplaying files. |
| `apollo-r3-judge-b.sh` | `apollo-judge.sh` | The eight Llama-2/Qwen2.5 roleplaying files. |
| `apollo-qwen38-ood-judge.sh` | `apollo-r4-judge.sh` | Qwen3.8 insider-trading response files. |
| `apollo-r5-judge.sh` | `apollo-r4-judge.sh` | The six Gemma extra-seed insider files. |
| `apollo-r7-judge.sh` | `apollo-r4-judge.sh` | Gemma-4-31B `ap_rand_s2_insider_trading_none.jsonl`. |
| `apollo-r8-judge.sh` | `apollo-r4-judge.sh` | Gemma-4-12B `ap_rand_s2_insider_trading_none.jsonl`. |
| `apollo-r3-judge-d.sh` | `apollo-r3-judge-c.sh` | Kimi `ap_rand_s1_roleplaying_none.jsonl`; retains the legacy thinking delimiters. |
| `code-judge-qwen38-val2.sh` | `code-judge-qwen38-val.sh` | The LoRA conflicting and base EvilGenie files, with `OUT_DIR=results/code_eval/judge_qwen38_v2`. |

For example, from the repository root, after adapting the retained wrapper's
runtime setup for the execution environment:

```bash
sbatch experiments/soo/slurm/apollo-r4-judge.sh \
  results/apollo_eval/gemma4_31b/ap_rand_s2_insider_trading_none.jsonl

OUT_DIR=results/code_eval/judge_qwen38_v2 \
sbatch experiments/soo/slurm/code-judge-qwen38-val.sh \
  results/code_eval/gemma4_12b/lora_s0_impossible_conflicting.jsonl \
  results/code_eval/gemma4_12b/base_evilgenie.jsonl
```

The replacements keep judge arguments and thinking settings, including both
the code-judge and call-out passes. They use the retained wrapper's Slurm resource
defaults; pass `sbatch --time=...` etc. when a different job allocation is needed.
Historical file lists and scheduling settings remain in the
[pre-cleanup Slurm directory](https://github.com/DRAGNLabs/self-concept-experiments/tree/d84856ccdde03188e975a47799135fe40be19966/experiments/soo/slurm).
To recover any old wrapper without changing the working tree:

```bash
git show d84856ccdde03188e975a47799135fe40be19966:experiments/soo/slurm/apollo-r7-judge.sh
```

Cleanup verification compared the actual shell job bodies with a recording
Python stub: all four original default command sets, all ten replacement command
sequences, and filenames containing spaces. No model was loaded and no job was
submitted. All remaining Slurm scripts passed `bash -n`, and retained executables
were scanned for references to the removed filenames.
