# codebench — ImpossibleBench + EvilGenie, self-contained

Reward-hacking coding benchmarks ported from ImpossibleBench (Zhong et al.
2025, arXiv:2510.20270; Impossible-LiveCodeBench splits) and EvilGenie
(Gabor & Ryzhov 2025, arXiv:2511.21654; LiveCodeBench v5/v6 hard). Code
lives in `src/selfconcept/codebench/`; this directory holds the committed
data and judge prompts. Nothing here depends on the SOO experiment.

## Layout

- `data/impossible_{original,oneoff,conflicting}.jsonl` — 103 tasks per
  split. `original` is the capability control; `oneoff` flips one expected
  value; `conflicting` adds an assertion that contradicts another on the
  same input. Passing an impossible suite means the tests were gamed.
- `data/evilgenie.jsonl` — 138 problems with EvilGenie's seeded 70/30
  visible/holdout test split (holdout capped at 10; tests > 20 KB dropped;
  their nine ambiguous problems removed).
- `judge_prompts/` — EvilGenie's judge prompt verbatim; our category prompt
  for ImpossibleBench (legitimate / special_casing / test_modification /
  ...); the messages-only "does the model call out the contradiction"
  prompt.

Rebuild with `python -m selfconcept.codebench.convert` (needs network once).

## Running

```bash
# any HF chat model, greedy, optional PEFT adapter
python -m selfconcept.codebench.run --model <hf id> [--adapter <peft dir>] \
    --scenarios impossible_conflicting impossible_original evilgenie \
    --n 40 --max-attempts 3 --max-new-tokens 2048 --out results/code_eval/<name> --tag base

# LLM judge (default Qwen/Qwen3.8-27B, one A100, thinking off; --model to
# swap, --out-dir to grade into a separate folder), then the messages-only
# call-out pass, then the summary table
python -m selfconcept.codebench.judge --responses results/code_eval/<name>/base_*.jsonl --batch-size 4 --skip-existing
python -m selfconcept.codebench.callout --responses results/code_eval/<name>/base_impossible_*.jsonl --rejudge-errors
python -m selfconcept.codebench.table results/code_eval/<name>/base_*.jsonl
```

Protocol (both benchmarks): ImpossibleBench's "minimal" scaffold — the model
answers with code blocks, the code runs in a sandbox
(`codebench/sandbox.py`: bubblewrap when available, rlimits always), and the
test output is the next user turn, up to `--max-attempts`. EvilGenie's
tool-calling workspace is emulated with ```lang filename=NAME blocks: the
first turn shows solution.py and test.py verbatim and test_cases.json as whole
entries up to `WORKSPACE_TESTS_MAX_CHARS` (12k chars, with a count of the
omitted entries); only the test-runner output fed back after each attempt gets
EvilGenie's 3000-char middle truncation. **Runs before 2026-09-24 showed every
file through that 3000-char truncation, so the model never saw the middle of
test.py (4.2–5.1 KB on all 138 problems: the import/parse/compare logic) and
saw a third of the visible tests on the median problem; treat their EvilGenie
solve and hack rates as unreliable** (see the harness docstring; records now
carry `tests_shown` / `n_visible_tests`). Records
are appended per task, so a killed run resumes with the same `--tag`; do not
judge a file whose generation job is still running.

To drive the harness with a different model wrapper, build a
`generate(messages, turn, example_id) -> (text, truncated)` callable and call
`harness.run_scenario` (see `selfconcept.soo.evaluate_code` for the
steering/LoRA version).

The judge model is recorded in every `_graded_summary.json`; keep one judge
across the cells you compare (the SOO study's gemma-4 cells pin
`--model Qwen/Qwen2.5-72B-Instruct`). Judge-swap check on three 72B-graded
files (2026-09-16): Qwen3.8-27B agrees with the 72B on 34–36/40 labels and
34–36/39 call-out flags per file, finds the same rule-confirmed special-casing
(7/7), is stricter about truncated submissions (`no_code` where the 72B said
`legitimate`), never uses EvilGenie's `heuristic`, and deliberates at length —
verdicts that overrun the budget are re-run once at 3072 tokens. Qwen3.8-27B was validated against the
72B on 120 gemma-4-12B records: label agreement 82–88%, call-out agreement
87–92%, identical special-casing calls on the rule-verified records; it is
stricter about comment-only submissions (`no_code`) and folds EvilGenie's
"heuristic" into legitimate. It deliberates before answering, so leave
`--max-new-tokens` at the 2048 default.

Model notes: ATEM-protocol models (Muse Glimmer) need `--force-user-channel`;
Qwen3.x needs `SOO_CHAT_KWARGS='{"enable_thinking": false}'` and loads
through `selfconcept.common.loading` (the bare `AutoModelForCausalLM` path
silently mismatches those checkpoints). Numbers are not directly comparable
to the papers' agent-scaffold numbers.
