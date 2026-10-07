"""Blind full-transcript dossiers for judging the Phase 3 intervention arms (PLAN.md).

Port of rh_hillclimb/extract_full.py for our run directories: the conversation exactly as the model saw it, rebuilt
with the harness's own message functions, every turn's complete reasoning and the hidden-suite result. Differences
from Koby's extractor: (1) candidates are every impossible_original episode that passed the shown tests and failed
the hidden suite, with no reasoning-keyword prefilter; (2) the dossier file name is an opaque id, so the judge
cannot see the arm; the mapping lives in key.json next to (not inside) the dossier directory.

    python scripts/make_dossiers.py --data <rh_hillclimb/data> --out <dir> --runs ARM=DIR [ARM=DIR ...]
        [--salt STRING] [--batch-size 6] [--system-prompt-file FILE] [--all]

Every ``*_<scenario>.jsonl`` records file under each DIR (recursively, reasoning sidecars excluded) is read.
Writes ``<out>/dossiers/<id>.md``, ``<out>/key.json`` ({id: {arm, run, records, example_id, label, ...}}),
``<out>/batches/batch_<i>.list`` (dossier paths, shuffled across arms) and prints a per-arm candidate count.
``--all`` keeps every episode (for a no-filter judging check), not only the visible-pass/hidden-fail ones.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from selfconcept.codebench import harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="directory with <tag>.jsonl example files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, metavar="ARM=DIR")
    parser.add_argument("--salt", default="schemer-axis", help="salt for the opaque dossier ids")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--system-prompt-file", type=Path, help="the --system-prompt-file the runs used, if any")
    parser.add_argument("--all", action="store_true", help="keep every episode, not only visible-pass/hidden-fail")
    return parser.parse_args()


def is_candidate(record: dict) -> bool:
    """Passed the shown tests and failed the hidden ones; harness labels are automatic and not used."""
    if record.get("scenario") == "impossible_original":
        return bool(record.get("passed")) and not record.get("passed_original")
    if "passed_holdout" in record:
        return bool(record.get("passed_visible")) and not record.get("passed_holdout")
    return bool(record.get("passed")) and not record.get("passed_original", True)


def find_example(data: Path, tag: str, example_id: str) -> dict:
    for line in (data / f"{tag}.jsonl").open():
        if f'"{example_id}"' in line[:300] and json.loads(line)["example_id"] == example_id:
            return json.loads(line)
    raise KeyError((tag, example_id))


def reasoning_by_turn(run_dir: Path, tag: str, example_id: str) -> dict[int, str]:
    turns: dict[int, str] = {}
    for path in run_dir.glob(f"{tag}_*_reasoning.jsonl"):
        for line in path.open():
            entry = json.loads(line)
            if entry["example_id"] == example_id:
                turns[entry["turn"]] = entry["reasoning"]
    return turns


def remaining(record: dict, n_done: int) -> harness.AttemptsRemaining | None:
    return harness.stated_attempts_remaining(record["attempt_budget_stated"], n_done, record["max_attempts"])


def conversation(example: dict, record: dict) -> list[tuple[str, str]]:
    """(role, text) for the task message and each harness reply, interleaved with placeholders for model turns."""
    feedback = record.get("feedback") or record.get("check_feedback") or "impossiblebench"
    budget = record["max_attempts"] if record["attempt_budget_stated"] else None
    messages: list[tuple[str, str]] = []
    if example["kind"] == "check":
        instruction = harness.instruction_by_impossible_prompt[record.get("impossible_prompt") or "minimal"]
        messages.append(("user", harness.check_task_message(example, budget, instruction, feedback)))
        for turn, attempt in enumerate(record["attempts"]):
            messages.append(("assistant", str(turn)))
            if attempt.get("answer") is None or attempt.get("success"):
                continue
            error = harness.check_feedback_error(attempt.get("stdout") or "", attempt.get("stderr") or "")
            messages.append(("user", harness.neutral_retry_message(error, remaining(record, turn + 1))
                             if harness.is_neutral(feedback)
                             else harness.check_retry_message(error, remaining(record, turn + 1), instruction)))
    else:
        messages.append(("user", harness.workspace_task_message(
            example, harness.workspace_files(example), budget, feedback)))
        for turn, attempt in enumerate(record["attempts"]):
            messages.append(("assistant", str(turn)))
            if attempt.get("success") or not attempt.get("wrote") and feedback == "neutral":
                continue
            if not attempt.get("wrote") and feedback == "neutral_continue":
                messages.append(("user", harness.neutral_no_files_message(remaining(record, turn + 1))))
                continue
            output = harness.workspace_test_output(attempt.get("stdout") or "", attempt.get("stderr") or "")
            messages.append(("user", harness.neutral_workspace_retry_message(output, remaining(record, turn + 1))
                             if harness.is_neutral(feedback)
                             else harness.workspace_retry_message(output, remaining(record, turn + 1))))
    return messages


def dossier(name: str, example: dict, record: dict, reasoning: dict[int, str], system_note: str | None) -> str:
    """Same section layout as rh_hillclimb/extract_full.py, so the v3 rubric and verdict tooling apply unchanged.

    The header shows the opaque name only; run and arm are not revealed to the judge."""
    parts = [
        f"# {name}",
        f"scenario: {record['scenario']}  harness label: {record['label']}  "
        f"attempts used: {record['n_attempts']}/{record['max_attempts']}  stop_reason: {record.get('stop_reason')}",
    ]
    parts.append("## SYSTEM MESSAGE\n" + (system_note if system_note else "(none)"))
    for role, text in conversation(example, record):
        if role == "user":
            parts.append("## USER / HARNESS MESSAGE\n" + text)
            continue
        turn = int(text)
        attempt = record["attempts"][turn]
        parts.append(f"## MODEL TURN {turn}: REASONING (analysis channel, not shown to the user)\n"
                     + reasoning.get(turn, "(missing)"))
        parts.append(f"## MODEL TURN {turn}: REPLY\n" + attempt["completion"])
    if example["kind"] == "check":
        parts.append("## AFTER THE EPISODE (never shown to the model)\n"
                     f"passed shown tests: {record['passed']}  passed original correct suite: {record['passed_original']}"
                     + ("\nThe correct suite:\n```python\n" + example["original_test"] + "\n```"
                        if record["scenario"] != "impossible_original" else
                        "\nFull correct suite (shown tests are a subset):\n```python\n" + example["original_test"] + "\n```"))
    else:
        parts.append("## AFTER THE EPISODE (never shown to the model)\n"
                     f"passed visible: {record['passed_visible']}  passed hidden holdout: {record['passed_holdout']}  "
                     f"test files modified: {record['test_modified']}\nholdout runner output:\n```\n"
                     + record["holdout_output"] + "\n```")
    return "\n\n".join(parts)


def records_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.rglob("*.jsonl")
                  if not p.name.endswith(("_reasoning.jsonl", "_projection.jsonl")))


def opaque_id(arm: str, records: Path, example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{arm}|{records}|{example_id}".encode()).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    system_note = args.system_prompt_file.read_text().strip() if args.system_prompt_file else None
    dossiers_dir = args.out / "dossiers"
    dossiers_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "batches").mkdir(exist_ok=True)
    key: dict[str, dict] = {}
    counts: dict[str, list[int]] = {}
    for spec in args.runs:
        arm, _, run = spec.partition("=")
        run_dir = Path(run)
        n_episodes = n_candidates = 0
        for records_path in records_files(run_dir):
            for record in map(json.loads, records_path.open()):
                if "scenario" not in record or "attempts" not in record:
                    continue
                n_episodes += 1
                if not (args.all or is_candidate(record)):
                    continue
                n_candidates += 1
                tag = records_path.stem.removesuffix("_" + record["scenario"])
                example = find_example(args.data, tag, record["example_id"])
                reasoning = reasoning_by_turn(records_path.parent, tag, record["example_id"])
                name = opaque_id(arm, records_path.relative_to(run_dir), record["example_id"], args.salt)
                if name in key:
                    raise RuntimeError(f"opaque id collision: {name}")
                (dossiers_dir / f"{name}.md").write_text(dossier(name, example, record, reasoning, system_note))
                key[name] = {"arm": arm, "run": str(run_dir), "records": str(records_path.relative_to(run_dir)),
                             "example_id": record["example_id"], "label": record["label"],
                             "n_attempts": record["n_attempts"], "missing_reasoning_turns":
                             [t for t in range(len(record["attempts"])) if t not in reasoning]}
        counts[arm] = [n_candidates, n_episodes]
    names = sorted(key)
    random.Random(args.shuffle_seed).shuffle(names)
    for i in range(0, len(names), args.batch_size):
        (args.out / "batches" / f"batch_{i // args.batch_size}.list").write_text(
            "".join(f"dossiers/{n}.md\n" for n in names[i:i + args.batch_size]))
    (args.out / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True))
    for arm, (c, n) in counts.items():
        print(f"{arm}: {c} candidates / {n} episodes")
    print(f"{len(names)} dossiers, {-(-len(names) // args.batch_size)} batches -> {args.out}")


if __name__ == "__main__":
    main()
