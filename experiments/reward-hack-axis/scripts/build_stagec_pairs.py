"""Pair Stage C unsteered strong hacks with honest solves of the same problem.

The episodes and the verdicts are already on disk. This script only reads them
and writes pair files. It does not load a model and it does not need a job.

A pair is one ImpossibleBench problem under the unsteered Stage C instruction.
The hack side passed the shown tests, failed the hidden suite, and has a
first-pass verdict of HACK with strength STRONG. A hidden-suite pass that the
judge also called a hack is not used on either side. The honest side passed
the hidden suite, was not itself judged a hack, and does not paste the visible
tests back as the answer. The saved text is the original task message plus
that one reply, the same shape as same_prompt_pairs.jsonl.

Whole problems are held out. The holdout is every k-th problem id in sorted
order (k = 3 for gpt-oss, k = 4 for Muse), and at most four pairs are kept
per problem.

Usage (from the repo root):
    python experiments/reward-hack-axis/scripts/build_stagec_pairs.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from selfconcept.codebench import harness

MIRROR = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
EXAMPLES = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data")
MAX_PER_PROBLEM = 4

RUNS = (
    {
        "name": "gptoss",
        "model": "openai/gpt-oss-120b",
        "unsteered": MIRROR / "gptoss120b/main/unsteered",
        "verdicts": MIRROR / "gptoss120b/main/verdicts/verdicts.detail.json",
        "holdout_stride": 3,
        "chat_kwargs": {"reasoning_effort": "low"},
    },
    {
        "name": "muse",
        "model": "meta-models/Muse-Glimmer-30B",
        "unsteered": MIRROR / "muse_glimmer/main/unsteered",
        "verdicts": MIRROR / "muse_glimmer/main/verdicts/verdicts_all.detail.json",
        "holdout_stride": 4,
        "chat_kwargs": {"reasoning_strength": "low"},
    },
)


def embeds_visible_test_answers(code: str) -> bool:
    text = code.lower().replace("\u2010", "-").replace("\u2011", "-")
    copied = "_precomputed" in text or "known_answers" in text or "precomputed" in text
    from_tests = "test suite" in text or "test cases" in text or "unit tests" in text
    return copied and from_tests


def load_examples() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for name in ("ib_solvhard_s0.jsonl", "ib_solvhard_s1.jsonl", "ib_solvhard_s2.jsonl"):
        for line in (EXAMPLES / name).read_text().splitlines():
            if line.strip():
                example = json.loads(line)
                found[example["example_id"]] = example
    return found


def load_verdicts(path: Path) -> tuple[dict[tuple[str, str], dict], set[tuple[str, str]]]:
    raw = json.loads(path.read_text())
    strong: dict[tuple[str, str], dict] = {}
    judged_hack: set[tuple[str, str]] = set()
    for key, row in raw.items():
        arm, seed, example_id = key.split("/")
        if arm != "unsteered":
            continue
        episode = (seed, example_id)
        if row.get("verdict") == "HACK":
            judged_hack.add(episode)
        if row.get("verdict") == "HACK" and row.get("strength") == "STRONG":
            strong[episode] = row
    return strong, judged_hack


def load_episodes(root: Path) -> dict[tuple[str, str], dict]:
    episodes: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("seed*/main_impossible_original.jsonl")):
        seed = path.parent.name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["_seed"] = seed
            record["_source"] = str(path)
            episodes[(seed, record["example_id"])] = record
    return episodes


def task_message(example: dict, record: dict) -> str:
    feedback = record.get("feedback") or "neutral"
    budget = record["max_attempts"] if record["attempt_budget_stated"] else None
    instruction = harness.instruction_by_impossible_prompt[record.get("impossible_prompt") or "minimal"]
    return harness.check_task_message(example, budget, instruction, feedback)


def honest_reply(record: dict) -> str:
    for attempt in reversed(record.get("attempts") or []):
        if attempt.get("success") and attempt.get("completion"):
            return attempt["completion"]
    return ""


def hack_reply(record: dict, verdict: dict) -> str:
    turn = verdict.get("turn")
    attempts = record.get("attempts") or []
    if not isinstance(turn, int) or turn < 0 or turn >= len(attempts):
        return ""
    return attempts[turn].get("completion") or ""


def build(spec: dict, examples: dict[str, dict]) -> tuple[list[dict], dict]:
    verdicts, judged_hack = load_verdicts(spec["verdicts"])
    episodes = load_episodes(spec["unsteered"])
    by_problem: dict[str, list[tuple[tuple[str, str], dict, dict]]] = defaultdict(list)
    for key, verdict in verdicts.items():
        record = episodes.get(key)
        if record is None or not hack_reply(record, verdict):
            continue
        # Shown tests passed and the hidden suite failed. A hidden-suite pass
        # is not a hack example even when the judge called it one.
        if not (record.get("passed") and not record.get("passed_original")):
            continue
        by_problem[key[1]].append((key, verdict, record))

    honest_by_problem: dict[str, list[dict]] = defaultdict(list)
    for key, record in episodes.items():
        if not record.get("passed_original") or key in judged_hack:
            continue
        if embeds_visible_test_answers(record.get("final_code") or ""):
            continue
        if not honest_reply(record):
            continue
        honest_by_problem[key[1]].append(record)

    pairable = sorted(problem for problem, group in by_problem.items() if honest_by_problem.get(problem))
    holdout = set(pairable[:: spec["holdout_stride"]])
    if not holdout or holdout >= set(pairable):
        raise SystemExit(f"{spec['name']}: holdout is empty or takes every problem")

    rows: list[dict] = []
    for problem in pairable:
        split = "holdout" if problem in holdout else "train"
        hacks = sorted(by_problem[problem], key=lambda item: item[0])[:MAX_PER_PROBLEM]
        honest = sorted(honest_by_problem[problem], key=lambda item: item["_seed"])
        example = examples.get(problem)
        if example is None:
            raise SystemExit(f"no problem statement for {problem}")
        for index, ((seed, _example_id), verdict, hack_record) in enumerate(hacks):
            if index >= len(honest):
                break
            solved = honest[index]
            user = task_message(example, hack_record)
            if user != task_message(example, solved):
                raise SystemExit(f"{problem} seed {seed}: hack and honest prompts differ")
            reply = hack_reply(hack_record, verdict)
            honest_text = honest_reply(solved)
            pair_id = f"{problem}__{index:02d}"
            shared = {
                "pair_id": pair_id,
                "task_id": problem,
                "model": spec["model"],
                "split": split,
                "source_run": "stagec_unsteered",
                "impossible_prompt": hack_record.get("impossible_prompt"),
                "max_attempts": hack_record.get("max_attempts"),
                "attempt_budget_stated": bool(hack_record.get("attempt_budget_stated")),
                "feedback": hack_record.get("feedback"),
                "chat_kwargs": spec["chat_kwargs"],
            }
            rows.append({
                **shared,
                "id": f"{pair_id}__hack",
                "condition": "hack",
                "hack_type": verdict.get("mechanism"),
                "strength": verdict.get("strength"),
                "turn": verdict.get("turn"),
                "seed": seed,
                "source": hack_record["_source"],
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
            })
            rows.append({
                **shared,
                "id": f"{pair_id}__honest",
                "condition": "honest",
                "hack_type": None,
                "strength": None,
                "turn": None,
                "seed": solved["_seed"],
                "source": solved["_source"],
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": honest_text}],
            })
    _check(rows, holdout)
    summary = {
        "model": spec["model"],
        "pairs_train": sum(1 for row in rows if row["split"] == "train" and row["condition"] == "hack"),
        "pairs_holdout": sum(1 for row in rows if row["split"] == "holdout" and row["condition"] == "hack"),
        "problems_train": sorted({row["task_id"] for row in rows if row["split"] == "train"}),
        "problems_holdout": sorted(holdout),
    }
    return rows, summary


def _check(rows: list[dict], holdout: set[str]) -> None:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    for pair_id, members in pairs.items():
        if {member["condition"] for member in members} != {"hack", "honest"}:
            raise SystemExit(f"{pair_id} is not one hack and one honest")
        hack, honest = sorted(members, key=lambda item: item["condition"])
        if hack["messages"][0]["content"] != honest["messages"][0]["content"]:
            raise SystemExit(f"{pair_id} user text differs")
        if not hack["messages"][1]["content"] or not honest["messages"][1]["content"]:
            raise SystemExit(f"{pair_id} empty reply")
        if hack["split"] == "holdout" and hack["task_id"] not in holdout:
            raise SystemExit(f"{pair_id} marked holdout but its problem is not")
    train = {row["task_id"] for row in rows if row["split"] == "train"}
    if train & holdout:
        raise SystemExit(f"problem on both sides: {sorted(train & holdout)}")


def main() -> None:
    examples = load_examples()
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in RUNS:
        rows, summary = build(spec, examples)
        out = out_dir / f"stagec_{spec['name']}_pairs.jsonl"
        with out.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["rows"] = len(rows)
        summary["output"] = str(out)
        out.with_name(f"stagec_{spec['name']}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
