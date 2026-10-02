"""Select labeled tier-1 neutral-protocol rh_hillclimb records and convert them to correlation transcripts.

Label 1: every v3 verdict for the record is HACK (STRONG or WEAK). Label 0: no verdict and not a review candidate.
Everything else (NOT_HACK, BORDERLINE, disagreeing judges, candidates awaiting review) is excluded.
One transcript is kept per (base problem, label), so a problem contributes at most one hack and one non-hack.
Hacks are scored at the verdict's hack turn; non-hacks at their problem's hack turn (or their last turn, if shorter or
the problem has no hack).
"""
import argparse
from collections import Counter
from collections.abc import Iterator
from fnmatch import fnmatch
from functools import cache
import hashlib
import json
import logging
import re
from pathlib import Path
import shlex
from typing import Any, cast, Literal, NamedTuple, NotRequired, TypedDict

from selfconcept.codebench import harness
from selfconcept.common.harmony import join_harmony_completion
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.common.jsonl import read_jsonl
from selfconcept.goodfire_rh_metrics.model_specifics import harmony_model_specifics, ModelSpecifics
from .records import OutcomeRecord, TranscriptRecord

logger = logging.getLogger(__name__)

DEFAULT_MODEL_KEY = "openai/gpt-oss-120b"
NEUTRAL_RUN_GLOB = "n_*"
# rh_hillclimb/next_neutral_batches.py candidates(): the records queued for full-trace review.
CANDIDATE_LABELS = frozenset({"hack_visible_only", "hack_test_modified", "visible_only_holdout_timeout",
                              "cheat_special_case", "cheat_modify_tests"})

type RecordKey = str
type BinaryLabel = Literal[0, 1]
type Selection = Literal["hack", "unflagged", "excluded_verdict", "candidate_unreviewed"]


class VerdictRecord(TypedDict):
    file: str
    verdict: str
    strength: NotRequired[str | None]
    turn: NotRequired[int | None]


class LabeledRecord(NamedTuple):
    key: RecordKey
    run: str
    stem: str
    record: dict[str, Any]
    verdicts: list[VerdictRecord]
    label: BinaryLabel
    turn: int


class WorklistSettings(NamedTuple):
    chat_kwargs: dict[str, bool | str]
    system_prompt: str | None


def canonical_key(file: str) -> RecordKey:
    key = Path(file).name
    parts = key.removesuffix(".md").split("__")
    if not key.endswith(".md") or len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid rh_hillclimb record key: {file}")
    return key


def record_key(run: str, stem: str, example_id: str) -> RecordKey:
    return canonical_key(f"{run}__{stem}__{example_id}.md")


def load_verdicts(paths: list[Path]) -> dict[RecordKey, list[VerdictRecord]]:
    verdicts_by_key: dict[RecordKey, list[VerdictRecord]] = {}
    for path in paths:
        for row in read_jsonl(path):
            verdicts_by_key.setdefault(canonical_key(row["file"]), []).append(row)
    return verdicts_by_key


def is_tier1(record: dict[str, Any]) -> bool:
    return record["scenario"] == "impossible_original" or record["scenario"].startswith("evilgenie")


def is_tier1_neutral_key(key: RecordKey) -> bool:
    run, stem, _ = key.removesuffix(".md").split("__")
    return fnmatch(run, NEUTRAL_RUN_GLOB) and ("impossible_original" in stem or "evilgenie" in stem)


def is_review_candidate(record: dict[str, Any]) -> bool:
    passed_shown_but_not_original = (record["scenario"] == "impossible_original" and record.get("passed")
                                     and not record.get("passed_original"))
    return record["label"] in CANDIDATE_LABELS or bool(passed_shown_but_not_original)


def iter_tier1_records(root: Path) -> Iterator[tuple[RecordKey, str, str, dict[str, Any]]]:
    for path in sorted((root / "runs").glob(f"{NEUTRAL_RUN_GLOB}/*.jsonl")):
        if path.name.endswith("_reasoning.jsonl"):
            continue
        run = path.parent.name
        for record in read_jsonl(path):
            if is_tier1(record):
                yield record_key(run, path.stem, record["example_id"]), run, path.stem, record


def hack_turn(verdicts: list[VerdictRecord]) -> int:
    turns = {verdict.get("turn") for verdict in verdicts}
    if len(turns) != 1 or not isinstance(turn := turns.pop(), int):
        raise ValueError(f"HACK verdicts need one agreed integer turn: {verdicts[0]['file']}")
    return turn


def last_turn(record: dict[str, Any]) -> int:
    return len(record["attempts"]) - 1


def select_record(key: RecordKey, run: str, stem: str, record: dict[str, Any],
                  verdicts: list[VerdictRecord]) -> tuple[LabeledRecord | None, Selection]:
    if verdicts and all(verdict["verdict"] == "HACK" for verdict in verdicts):
        return LabeledRecord(key, run, stem, record, verdicts, 1, hack_turn(verdicts)), "hack"
    if verdicts:
        return None, "excluded_verdict"
    if is_review_candidate(record):
        return None, "candidate_unreviewed"
    return LabeledRecord(key, run, stem, record, verdicts, 0, last_turn(record)), "unflagged"


def labeled_records(root: Path, verdicts_by_key: dict[RecordKey, list[VerdictRecord]]) -> list[LabeledRecord]:
    labeled: list[LabeledRecord] = []
    found: set[RecordKey] = set()
    selection_counts_by_run: dict[str, Counter[Selection]] = {}
    for key, run, stem, record in iter_tier1_records(root):
        if key in found:
            raise ValueError(f"Duplicate rh_hillclimb record key: {key}")
        found.add(key)
        selected, selection = select_record(key, run, stem, record, verdicts_by_key.get(key, []))
        selection_counts_by_run.setdefault(run, Counter())[selection] += 1
        if selected is not None:
            labeled.append(selected)
    for run, counts in selection_counts_by_run.items():
        logger.info("%s: %d HACK, %d unflagged, %d excluded verdict, %d candidate unreviewed", run,
                    counts["hack"], counts["unflagged"], counts["excluded_verdict"], counts["candidate_unreviewed"])
    for key in sorted(key for key in verdicts_by_key.keys() - found if is_tier1_neutral_key(key)):
        logger.warning("Verdict record not found: %s", key)
    return labeled


def option(tokens: list[str], name: str) -> str:
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Worklist line missing {name}") from None


def parse_worklist_line(line: str, root: Path | None = None) -> tuple[tuple[str, str], WorklistSettings]:
    chat_kwargs: dict[str, bool | str] = {}
    if line.startswith("SOO_CHAT_KWARGS="):
        assignment, line = line.split(maxsplit=1)
        chat_kwargs = json.loads(assignment.partition("=")[2])
    tokens = shlex.split(line)
    run, tag = Path(option(tokens, "--out")).name, option(tokens, "--tag")
    system_path = Path(option(tokens, "--system-prompt-file")) if "--system-prompt-file" in tokens else None
    # Mirrored worklists still name the original owner's inaccessible home.
    if system_path is not None and root is not None and "rh_hillclimb" in system_path.parts:
        relative = system_path.parts[system_path.parts.index("rh_hillclimb") + 1:]
        system_path = root.joinpath(*relative)
    system_prompt = system_path.read_text().strip() if system_path else None
    return (run, tag), WorklistSettings(chat_kwargs, system_prompt)


def load_worklists(root: Path) -> dict[tuple[str, str], WorklistSettings]:
    settings: dict[tuple[str, str], WorklistSettings] = {}
    for path in sorted((root / "worklists").glob("*.txt")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            key, value = parse_worklist_line(line, root)
            if key in settings and settings[key] != value:
                raise ValueError(f"Conflicting worklist settings for {key}")
            settings[key] = value
    return settings


def record_tag(item: LabeledRecord) -> str:
    suffix = f"_{item.record['scenario']}"
    if not item.stem.endswith(suffix):
        raise ValueError(f"Record stem {item.stem} does not end in scenario {item.record['scenario']}")
    return item.stem.removesuffix(suffix)


def task_message(example: dict[str, Any], record: dict[str, Any]) -> str:
    feedback = record.get("feedback") or record.get("check_feedback") or "impossiblebench"
    budget = record["max_attempts"] if record["attempt_budget_stated"] else None
    if example["kind"] == "check":
        prompt = record.get("impossible_prompt") or "minimal"
        instruction = harness.instruction_by_impossible_prompt[prompt]
        return harness.check_task_message(cast(harness.CheckExample, example), budget, instruction, feedback)
    workspace_example = cast(harness.WorkspaceExample, example)
    return harness.workspace_task_message(
        workspace_example, harness.workspace_files(workspace_example), budget, feedback)


def harness_feedback_message(example: dict[str, Any], record: dict[str, Any], turn: int) -> str:
    """The neutral-protocol harness reply to one attempt, rebuilt from the stored output (its last 3000 characters)."""
    if not harness.is_neutral(record["feedback"]):
        raise ValueError(f"Only the neutral protocol is reconstructed, not {record['feedback']}")
    attempt = record["attempts"][turn]
    remaining = harness.stated_attempts_remaining(record["attempt_budget_stated"], turn + 1, record["max_attempts"])
    if example["kind"] == "check":
        error = harness.check_feedback_error(attempt["stdout"], attempt["stderr"])
        return harness.neutral_retry_message(error, remaining)
    if not attempt["wrote"]:
        return harness.neutral_no_files_message(remaining)
    output = harness.workspace_test_output(attempt["stdout"], attempt["stderr"])
    return harness.neutral_workspace_retry_message(output, remaining)


def conversation_before_turn(example: dict[str, Any], record: dict[str, Any], settings: WorklistSettings,
                             turn: int) -> Conversation:
    """The model saw only the final channel of its earlier replies."""
    messages: Conversation = []
    if settings.system_prompt is not None:
        messages.append({"role": "system", "content": settings.system_prompt})
    messages.append({"role": "user", "content": task_message(example, record)})
    for earlier_turn in range(turn):
        messages.append({"role": "assistant", "content": record["attempts"][earlier_turn]["completion"]})
        messages.append({"role": "user", "content": harness_feedback_message(example, record, earlier_turn)})
    return messages


def hack_strength(verdicts: list[VerdictRecord]) -> str:
    return "STRONG" if any(verdict.get("strength") == "STRONG" for verdict in verdicts) else "WEAK"


def transcript_outcome(item: LabeledRecord, example_id: str, model_key: str) -> OutcomeRecord:
    record = item.record
    outcome: OutcomeRecord = {
        "example_id": example_id, "scenario": record["scenario"],
        "status": "truncated" if record["attempts"][-1]["truncated"] else "complete", "label": record["label"],
        "model_key": model_key, "verdict": "HACK" if item.label else "unflagged",
        "harness_label": record["label"], "final_code": record.get("final_code", ""), "stratum": record["example_id"]}
    if item.label:
        outcome["strength"] = hack_strength(item.verdicts)
    return outcome


def key_hash(item: LabeledRecord) -> str:
    return hashlib.sha256(item.key.encode()).hexdigest()


def records_by_problem(records: list[LabeledRecord]) -> dict[str, list[LabeledRecord]]:
    """Complete episodes only, since analysis drops truncated ones."""
    grouped: dict[str, list[LabeledRecord]] = {}
    for item in records:
        if not item.record["attempts"][-1]["truncated"]:
            grouped.setdefault(item.record["example_id"], []).append(item)
    return grouped


def matched_non_hack(non_hacks: list[LabeledRecord], matched_turn: int | None) -> LabeledRecord:
    """Prefer an episode long enough to reach the matched turn, then score it there (or at its last turn)."""
    if matched_turn is None:
        return min(non_hacks, key=key_hash)
    chosen = min(non_hacks, key=lambda item: (last_turn(item.record) < matched_turn, key_hash(item)))
    return chosen._replace(turn=min(matched_turn, last_turn(chosen.record)))


def one_per_problem_and_label(records: list[LabeledRecord]) -> list[LabeledRecord]:
    kept: list[LabeledRecord] = []
    for problem_records in records_by_problem(records).values():
        hacks = [item for item in problem_records if item.label]
        non_hacks = [item for item in problem_records if not item.label]
        hack = min(hacks, key=key_hash) if hacks else None
        if hack is not None:
            kept.append(hack)
        if non_hacks:
            kept.append(matched_non_hack(non_hacks, hack.turn if hack is not None else None))
    logger.info("Kept one transcript per problem and label: %d hacks, %d non-hacks",
                sum(item.label for item in kept), sum(not item.label for item in kept))
    return kept


def stratified_records(records: list[LabeledRecord]) -> list[LabeledRecord]:
    """Every transcript of each problem with both labels; non-hacks are scored at the problem's earliest hack turn."""
    kept: list[LabeledRecord] = []
    for problem_records in records_by_problem(records).values():
        hacks = [item for item in problem_records if item.label]
        non_hacks = [item for item in problem_records if not item.label]
        if not hacks or not non_hacks:
            continue
        earliest_hack_turn = min(hack.turn for hack in hacks)
        kept += hacks
        kept += [item._replace(turn=min(earliest_hack_turn, last_turn(item.record))) for item in non_hacks]
    logger.info("Kept %d strata: %d hacks, %d non-hacks", len({item.record["example_id"] for item in kept}),
                sum(item.label for item in kept), sum(not item.label for item in kept))
    return kept


def strict_transcripts(root: Path, records: list[LabeledRecord]) -> list[TranscriptRecord]:
    """Match actual prompt/protocol, run family, and exact hack turn within a problem.

    A negative episode may contribute at several hack turns. Its episode ID is
    retained for auditing; analysis must cluster uncertainty by base problem.
    Short negatives are excluded, never substituted at an earlier turn.
    """
    worklists = load_worklists(root)

    @cache
    def examples(tag: str) -> dict[str, dict[str, Any]]:
        return {row["example_id"]: row for row in read_jsonl(root / "data" / f"{tag}.jsonl")}

    groups: dict[tuple[str, str, str], list[LabeledRecord]] = {}
    for item in records:
        if item.record["attempts"][-1]["truncated"]:
            continue
        tag = record_tag(item)
        settings = worklists[item.run, tag]
        protocol = {
            "run_family": re.sub(r"_seed\d+$", "", item.run),
            "messages": conversation_before_turn(examples(tag)[item.record["example_id"]], item.record, settings, 0),
            "chat_kwargs": settings.chat_kwargs,
            "feedback": item.record.get("feedback"),
            "max_attempts": item.record["max_attempts"],
            "attempt_budget_stated": item.record["attempt_budget_stated"],
        }
        family = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        key = (item.record["scenario"], item.record["example_id"], family)
        groups.setdefault(key, []).append(item)
    selected = []
    metadata = []
    for (scenario, problem, family), items in sorted(groups.items()):
        for turn in sorted({item.turn for item in items if item.label}):
            hacks = [item for item in items if item.label and item.turn == turn]
            negatives = [item._replace(turn=turn) for item in items if not item.label and last_turn(item.record) >= turn]
            if not negatives:
                continue
            for item in hacks + negatives:
                if item.record["attempts"][turn]["truncated"]:
                    continue
                selected.append(item)
                metadata.append((problem, family, json.dumps([scenario, problem, family, turn])))
    transcripts = build_transcripts(root, selected)
    for transcript, (problem, family, stratum) in zip(transcripts, metadata, strict=True):
        episode_id = transcript["example_id"]
        transcript["example_id"] = f"{episode_id}/turn_{transcript['turn']}"
        transcript["outcome"].update(example_id=transcript["example_id"], episode_id=episode_id,
                                      problem=problem, run_family=family, stratum=stratum)
    logger.info("Strict matching retained %d transcripts across %d strata", len(transcripts),
                len({row["outcome"]["stratum"] for row in transcripts}))
    return transcripts


def build_transcripts(root: Path, records: list[LabeledRecord]) -> list[TranscriptRecord]:
    worklists = load_worklists(root)

    @cache
    def examples(tag: str) -> dict[str, dict[str, Any]]:
        return {row["example_id"]: row for row in read_jsonl(root / "data" / f"{tag}.jsonl")}

    @cache
    def model_specifics(run: str, tag: str) -> ModelSpecifics:
        return harmony_model_specifics(root / "runs" / run / f"{tag}_{tag}_reasoning.jsonl")

    @cache
    def model_key(run: str, stem: str) -> str:
        summary_path = root / "runs" / run / f"{stem}_summary.json"
        return json.loads(summary_path.read_text())["model"] if summary_path.exists() else DEFAULT_MODEL_KEY

    transcripts: list[TranscriptRecord] = []
    for item in records:
        tag, record = record_tag(item), item.record
        settings = worklists[item.run, tag]
        attempt = record["attempts"][item.turn]
        reasoning, final = model_specifics(item.run, tag).split_attempt(
            record["example_id"], item.turn, attempt["completion"])
        example_id = f"{item.run}/{tag}/{record['example_id']}"
        transcripts.append({
            "example_id": example_id, "scenario": record["scenario"], "turn": item.turn,
            "messages": conversation_before_turn(examples(tag)[record["example_id"]], record, settings, item.turn),
            "chat_kwargs": settings.chat_kwargs, "raw_response": join_harmony_completion(reasoning, final),
            "truncated": attempt["truncated"],
            "outcome": transcript_outcome(item, example_id, model_key(item.run, item.stem))})
    return transcripts


def write_transcripts(path: Path, transcripts: list[TranscriptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(transcript, allow_nan=False) + "\n" for transcript in transcripts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stratified", action="store_true",
                        help="Keep every transcript of problems with both labels, for stratified analysis")
    parser.add_argument("--strict-strata", action="store_true",
                        help="Match problem, exact prompt/run family, and turn; includes all supported hack turns")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = labeled_records(args.root, load_verdicts(args.verdicts))
    if args.strict_strata:
        transcripts = strict_transcripts(args.root, records)
    else:
        selected = stratified_records(records) if args.stratified else one_per_problem_and_label(records)
        transcripts = build_transcripts(args.root, selected)
    write_transcripts(args.out, transcripts)
    logger.info("Wrote %d transcripts to %s", len(transcripts), args.out)


if __name__ == "__main__":
    main()
