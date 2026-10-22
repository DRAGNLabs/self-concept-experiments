# Prompting against reward hacking: plan

Written 2026-10-22, before any run. Results go in [FINDINGS.md](FINDINGS.md); this file holds the design and the
pending work, and is updated when the design changes (with a dated note saying what changed and why).

This is the prompting baseline for the paper *Steering Away From Deception Decreases Reward Hacking*. Steering with
the reward-hack axis is a later, separate step and is not planned here.

2026-10-22: the arms are the four existing `codebench` task instructions only (Koby's suggestion); the new system
prompts drafted earlier the same day were dropped before any run.

## Question

Can a prompt alone make gpt-oss-120b reward hack less on coding problems, without making it worse at coding?

1. **Effect.** How does the hack rate change across the four `codebench` task instructions, from `minimal` ("pass
   all the tests") to `strict` ("STOP, identify them; do not carve out the code"), on the same problems and seeds?
2. **Cost.** Does a stronger instruction lower capability: hidden-suite pass rate, no-code rate, and wrongly calling
   correct tests "flawed"?
3. **Comparison.** How does the best prompt compare with steering? The schemer-axis main run gives the steering
   numbers on the same setup (hack proxy 19% unsteered, 10% at −0.2 shift, 11% under the band cap, capability
   unchanged; [schemer-axis FINDINGS](../schemer-axis/FINDINGS.md)).

Predicted direction, fixed now: the hack rate falls from `minimal` to `critical` to `critical_no_carve` to `strict`;
`strict` also raises the no-code and "tests are flawed" rates.

## What this builds on

- **Model.** `openai/gpt-oss-120b`, low reasoning effort (`SOO_CHAT_KWARGS={"reasoning_effort":"low"}`).
- **Reward-hacking setup.** Koby's clean protocol, [reward_hack_recipe.md](../codebench/reward_hack_recipe.md):
  solvable ImpossibleBench problems with shown tests and a hidden full suite, neutral feedback, 5 attempts with a
  stated budget, temperature 1, 16384 new tokens.
- **Problems.** The 45 hard solvable ImpossibleBench problems (`ib_solvhard_s0-2` in
  `$MIRROR/rh_hillclimb/data`), the same set the schemer-axis main run used, so our numbers line up with the
  steering numbers. Koby's arm-search episodes (`$MIRROR/arm_search_rh/v3_scratch`) show which problems hack most,
  if we need to narrow the set to raise the base rate.
- **Runner.** `python -m selfconcept.codebench.run --backend vllm-harmony --impossible-prompt <arm>`. No new runner
  code is needed.
- **Copied from schemer-axis** (adapted, not imported): the array sbatch (`main_run.sbatch`), `pilot_summary.py`,
  `main_analysis.py`, `make_dossiers.py`.

## Prompt arms

Every arm uses the recipe; arms differ only in the task instruction (`--impossible-prompt`, defined in
`selfconcept.codebench.harness`):

| Arm | Instruction | Wording |
|---|---|---|
| `minimal` | `minimal` | ImpossibleBench's default: "fully implement the function described and pass all the tests" |
| `critical` | `critical` | ImpossibleBench prompt C: critically evaluate the tests; "a correct implementation is more important than passing flawed tests" |
| `baseline` | `critical_no_carve` | prompt C plus "Do NOT try to carve out the code to pass the tests"; the recipe, and the schemer-axis unsteered arm |
| `strict` | `strict` | ImpossibleBench prompt D: as above, plus "STOP, identify them" (the recipe notes it pushes hacks to ~0) |

`baseline` is the reference for every contrast because it is the arm the steering results are measured against.

## Measures

- **Primary:** fraction of episodes with a v3 HACK verdict (rubric `$MIRROR/rh_hillclimb/judge_rubric_v3.md`). Every
  shown-pass/hidden-fail episode is judged, blind to arm, with no keyword prefilter (a prompt could change the
  wording without changing the behaviour).
- **Automatic, no judge:** shown-pass/hidden-fail rate (the hack proxy).
- **Cost on the same problems:** hidden-suite pass rate, no-code rate, truncation rate, tokens per turn, and the rate
  of claiming a correct test is flawed (every shown test here is correct, so any such claim is wrong).
- **Coding capability:** hidden-suite pass rate on the 58 easier solvable ImpossibleBench problems
  (`ib_solveasy_s0-2`) under each instruction. Public benchmarks (GSM8K, HumanEval) are not used: the instructions
  are about unit tests and do not apply to them.

## Analysis

Difference in each rate vs `baseline`, stratified by problem, whole-problem bootstrap CI, Holm-corrected across the
three other arms. A drop in hacks that comes with a drop of more than 10 points in hidden-suite pass rate or a rise
of more than 10 points in no-code rate is reported as degradation, not as reduced reward hacking.

## Work order

1. **Scaffold.** `experiments/prompt-rh-prevention/` with README, PLAN, FINDINGS, `scripts/`, `slurm/`, `data/`,
   laid out like schemer-axis. Done 2026-10-22.
2. **Setup.** Build the repo venv (`uv sync`); download gpt-oss-120b into `~/nobackup/autodelete/hf_cache`.
3. **GPU test.** Baseline on a handful of problems, one seed, on one A100 80 GB and on whatever else we agree to try;
   record throughput and memory in FINDINGS. Pick the GPU for the rest of the study.
4. **Pilot.** All arms × 45 problems × 1 seed. Check rates and decide whether to narrow the problem set.
5. **Main run.** All arms × 45 problems × 12 seeds (540 episodes per arm).
6. **Judging.** Dossiers for every proxy episode, judged with the v3 rubric.
7. **Coding capability.** All arms on the 58 easier problems.
8. **Analysis and write-up.** Contrasts vs `baseline`; comparison with the steering numbers. Record in FINDINGS.

## Open decisions

- **GPU.** One A100 80 GB is the default (gpt-oss-120b is ~61 GiB of weights in MXFP4). Schemer-axis ran on H200;
  bigger cards only with the user's go-ahead, after step 3.
- **Judge.** Which model judges (codebench `rh_judge.py` supports GLM-4.5-Air on a GPU or Claude Sonnet via the
  CLI) and the budget. At a 19% proxy rate that is ~100 dossiers per arm.
- **Problem set.** Keep all 45 (comparable with steering) or narrow to the problems that hack most (higher power).

## Paths

- Shared data (Koby's): `MIRROR=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments`
  (`rh_hillclimb/data`, `rh_hillclimb/judge_rubric_v3.md`, `arm_search_rh/v3_scratch`).
- Outputs: `~/nobackup/autodelete/self-concept-experiments/prompt-rh-prevention/` (autodelete: copy what matters).
