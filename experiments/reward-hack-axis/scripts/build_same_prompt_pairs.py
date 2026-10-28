"""Pair a hack sample with a non-hack sample of the same prompt.

A prompt is the same LiveCodeBench problem under the same instruction
(impossible_prompt, attempt budget, max attempts, neutral feedback, and
whether reasoning effort was low). One side is a v3 STRONG hack. The other
passed the hidden suite, was not itself judged a hack, and does not contain
a copied test-answer table.

Three problems are held out whole.

Usage (from the repo root):
    python experiments/reward-hack-axis/scripts/build_same_prompt_pairs.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HEADING = re.compile(r"^## (.+)$", re.M)
HOLDOUT = ("lcbhard_18", "lcbhard_22", "lcbhard_28")
MAX_PER_PROBLEM = 4


def embeds_visible_test_answers(code: str) -> bool:
    text = code.lower().replace("\u2010", "-").replace("\u2011", "-")
    copied = "_precomputed" in text or "known_answers" in text or "precomputed" in text
    from_tests = "test suite" in text or "test cases" in text or "unit tests" in text
    return copied and from_tests


def sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING.finditer(text))
    found = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found.append((match.group(1).strip(), text[start:end].strip()))
    return found


def section(text: str, title: str) -> str | None:
    for heading, body in sections(text):
        if heading == title:
            return body
    return None


def config_key(record: dict) -> tuple:
    return (
        record["example_id"],
        record.get("impossible_prompt"),
        record.get("max_attempts"),
        bool(record.get("attempt_budget_stated")),
        record.get("feedback"),
        "low" in record["_run"],
    )


def load_hacks(verdicts: Path) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for path in sorted(verdicts.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = Path(row["file"]).name
            if not name.startswith("n_") or "impossible_original" not in name:
                continue
            if row.get("verdict") == "HACK" and row.get("strength") == "STRONG":
                chosen[name] = row
    return chosen


def index_runs(runs: Path) -> dict[tuple[str, str], dict]:
    episodes: dict[tuple[str, str], dict] = {}
    for run_dir in sorted(runs.glob("n_*solv*")):
        for path in sorted(run_dir.glob("*.jsonl")):
            if path.name.endswith("_reasoning.jsonl"):
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("scenario") != "impossible_original":
                    continue
                record["_run"] = run_dir.name
                record["_source"] = f"runs/{run_dir.name}/{path.name}"
                episodes[(run_dir.name, record["example_id"])] = record
    return episodes


def honest_reply(record: dict) -> str:
    for attempt in reversed(record.get("attempts") or []):
        if attempt.get("success") and attempt.get("completion"):
            return attempt["completion"]
    attempts = record.get("attempts") or []
    if attempts and attempts[-1].get("completion"):
        return attempts[-1]["completion"]
    return record.get("final_code") or ""


def build(rh: Path) -> tuple[list[dict], dict]:
    hacks = load_hacks(rh / "verdicts" / "v3")
    episodes = index_runs(rh / "runs")
    hack_episode_ids = set()
    hack_groups: dict[tuple, list[tuple[str, dict, dict]]] = defaultdict(list)
    for name, verdict in hacks.items():
        run, _middle, example = name.removesuffix(".md").split("__")
        record = episodes.get((run, example))
        if record is None:
            continue
        hack_episode_ids.add((run, example))
        hack_groups[config_key(record)].append((name, verdict, record))

    solved_groups: dict[tuple, list[dict]] = defaultdict(list)
    for (run, example), record in episodes.items():
        if record.get("label") != "solved" or not record.get("passed_original"):
            continue
        if record.get("redefines_check") or (run, example) in hack_episode_ids:
            continue
        if embeds_visible_test_answers(record.get("final_code") or ""):
            continue
        solved_groups[config_key(record)].append(record)

    by_problem: dict[str, list[tuple[str, dict, dict, dict]]] = defaultdict(list)
    for key, group in hack_groups.items():
        pool = sorted(solved_groups.get(key, []), key=lambda item: item["_run"])
        for (name, verdict, hack_record), solved in zip(sorted(group, key=lambda item: item[0]), pool):
            by_problem[key[0]].append((name, verdict, hack_record, solved))

    missing = [problem for problem in HOLDOUT if problem not in by_problem]
    if missing:
        raise SystemExit(f"holdout problem has no pair: {missing}")

    rows: list[dict] = []
    for problem in sorted(by_problem):
        split = "holdout" if problem in HOLDOUT else "train"
        chosen = sorted(by_problem[problem], key=lambda item: item[0])[:MAX_PER_PROBLEM]
        for index, (name, verdict, hack_record, solved) in enumerate(chosen):
            dossier = (rh / "fulln" / name).read_text()
            user = section(dossier, "USER / HARNESS MESSAGE")
            turn = verdict.get("turn")
            reply = section(dossier, f"MODEL TURN {turn}: REPLY") if isinstance(turn, int) else None
            honest = honest_reply(solved)
            if not user or not reply or not honest:
                continue
            pair_id = f"{problem}__{index:02d}"
            shared = {
                "pair_id": pair_id,
                "task_id": problem,
                "template_id": problem,
                "split": split,
                "impossible_prompt": hack_record.get("impossible_prompt"),
                "max_attempts": hack_record.get("max_attempts"),
                "attempt_budget_stated": bool(hack_record.get("attempt_budget_stated")),
                "feedback": hack_record.get("feedback"),
                "reasoning_effort": "low" if "low" in hack_record["_run"] else "default",
            }
            rows.append({
                **shared,
                "id": f"{pair_id}__hack",
                "condition": "hack",
                "hack_type": verdict.get("mechanism"),
                "proxy_score": 1.0 if hack_record.get("passed") else 0.0,
                "true_score": 1.0 if hack_record.get("passed_original") else 0.0,
                "source": f"fulln/{name}",
                "run": hack_record["_run"],
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
            })
            rows.append({
                **shared,
                "id": f"{pair_id}__honest",
                "condition": "honest",
                "hack_type": None,
                "proxy_score": 1.0,
                "true_score": 1.0,
                "source": solved["_source"],
                "run": solved["_run"],
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": honest}],
            })
    _check(rows)
    summary = {
        "pairs_train": sum(1 for row in rows if row["split"] == "train" and row["condition"] == "hack"),
        "pairs_holdout": sum(1 for row in rows if row["split"] == "holdout" and row["condition"] == "hack"),
        "problems_train": sorted({row["task_id"] for row in rows if row["split"] == "train"}),
        "problems_holdout": list(HOLDOUT),
    }
    return rows, summary


def _check(rows: list[dict]) -> None:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    for pair_id, members in pairs.items():
        if {member["condition"] for member in members} != {"hack", "honest"}:
            raise SystemExit(f"{pair_id} is not one hack and one honest")
        hack, honest = sorted(members, key=lambda item: item["condition"])
        if hack["messages"][0]["content"] != honest["messages"][0]["content"]:
            raise SystemExit(f"{pair_id} user text differs")
        for field in ("impossible_prompt", "max_attempts", "attempt_budget_stated", "feedback", "reasoning_effort", "task_id", "split"):
            if hack[field] != honest[field]:
                raise SystemExit(f"{pair_id} {field} differs")
        if not hack["messages"][1]["content"] or not honest["messages"][1]["content"]:
            raise SystemExit(f"{pair_id} empty reply")
    train = {row["task_id"] for row in rows if row["split"] == "train"}
    holdout = {row["task_id"] for row in rows if row["split"] == "holdout"}
    if train & holdout:
        raise SystemExit(f"problem on both sides: {sorted(train & holdout)}")


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    rh = Path("/home/heychris/groups/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb")
    rows, summary = build(rh)
    out = repo / "experiments/reward-hack-axis/data/same_prompt_pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows"] = len(rows)
    summary["output"] = str(out)
    out.with_name("same_prompt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
