"""Select labeled tier-1 neutral-protocol rh_hillclimb records and convert them to correlation transcripts.

Label 1: every v3 verdict for the record is HACK (STRONG or WEAK). Label 0: no verdict and not a review candidate.
Everything else (NOT_HACK, BORDERLINE, disagreeing judges, candidates awaiting review) is excluded.
One transcript is kept per (base problem, label), so a problem contributes at most one hack and one non-hack.
"""
import argparse
from collections import Counter
from collections.abc import Iterator
from fnmatch import fnmatch
from functools import cache
import hashlib
import json
import logging
from pathlib import Path
import shlex
from typing import Any, cast, Literal, NamedTuple, NotRequired, TypedDict

from selfconcept.codebench import harness
from selfconcept.codebench.vllm_harmony import join_harmony_completion
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


class LabeledRecord(NamedTuple):
    key: RecordKey
    run: str
    stem: str
    record: dict[str, Any]
    verdicts: list[VerdictRecord]
    label: BinaryLabel


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


def select_record(key: RecordKey, run: str, stem: str, record: dict[str, Any],
                  verdicts: list[VerdictRecord]) -> tuple[LabeledRecord | None, Selection]:
    if verdicts and all(verdict["verdict"] == "HACK" for verdict in verdicts):
        return LabeledRecord(key, run, stem, record, verdicts, 1), "hack"
    if verdicts:
        return None, "excluded_verdict"
    if is_review_candidate(record):
        return None, "candidate_unreviewed"
    return LabeledRecord(key, run, stem, record, verdicts, 0), "unflagged"


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


def parse_worklist_line(line: str) -> tuple[tuple[str, str], WorklistSettings]:
    chat_kwargs: dict[str, bool | str] = {}
    if line.startswith("SOO_CHAT_KWARGS="):
        assignment, line = line.split(maxsplit=1)
        chat_kwargs = json.loads(assignment.partition("=")[2])
    tokens = shlex.split(line)
    run, tag = Path(option(tokens, "--out")).name, option(tokens, "--tag")
    system_path = Path(option(tokens, "--system-prompt-file")) if "--system-prompt-file" in tokens else None
    system_prompt = system_path.read_text().strip() if system_path else None
    return (run, tag), WorklistSettings(chat_kwargs, system_prompt)


def load_worklists(root: Path) -> dict[tuple[str, str], WorklistSettings]:
    settings: dict[tuple[str, str], WorklistSettings] = {}
    for path in sorted((root / "worklists").glob("*.txt")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            key, value = parse_worklist_line(line)
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


def initial_messages(example: dict[str, Any], record: dict[str, Any], settings: WorklistSettings) -> Conversation:
    messages: Conversation = []
    if settings.system_prompt is not None:
        messages.append({"role": "system", "content": settings.system_prompt})
    messages.append({"role": "user", "content": task_message(example, record)})
    return messages


def hack_strength(verdicts: list[VerdictRecord]) -> str:
    return "STRONG" if any(verdict.get("strength") == "STRONG" for verdict in verdicts) else "WEAK"


def transcript_outcome(item: LabeledRecord, example_id: str, model_key: str) -> OutcomeRecord:
    record = item.record
    outcome: OutcomeRecord = {
        "example_id": example_id, "scenario": record["scenario"],
        "status": "truncated" if record["attempts"][-1]["truncated"] else "complete", "label": record["label"],
        "model_key": model_key, "verdict": "HACK" if item.label else "unflagged",
        "harness_label": record["label"], "final_code": record.get("final_code", "")}
    if item.label:
        outcome["strength"] = hack_strength(item.verdicts)
    return outcome


def one_per_problem_and_label(records: list[LabeledRecord]) -> list[LabeledRecord]:
    """Choose among complete episodes only, since analysis drops truncated ones."""
    records_by_problem_and_label: dict[tuple[str, BinaryLabel], list[LabeledRecord]] = {}
    for item in records:
        if not item.record["attempts"][-1]["truncated"]:
            records_by_problem_and_label.setdefault((item.record["example_id"], item.label), []).append(item)
    kept = [min(group, key=lambda item: hashlib.sha256(item.key.encode()).hexdigest())
            for group in records_by_problem_and_label.values()]
    logger.info("Kept one transcript per problem and label: %d hacks, %d non-hacks",
                sum(item.label for item in kept), sum(not item.label for item in kept))
    return kept


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
        attempt = record["attempts"][0]
        reasoning, final = model_specifics(item.run, tag).split_attempt(record["example_id"], 0, attempt["completion"])
        example_id = f"{item.run}/{tag}/{record['example_id']}"
        transcripts.append({
            "example_id": example_id, "scenario": record["scenario"], "turn": 0,
            "messages": initial_messages(examples(tag)[record["example_id"]], record, settings),
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    selected = one_per_problem_and_label(labeled_records(args.root, load_verdicts(args.verdicts)))
    transcripts = build_transcripts(args.root, selected)
    write_transcripts(args.out, transcripts)
    logger.info("Wrote %d transcripts to %s", len(transcripts), args.out)


if __name__ == "__main__":
    main()
