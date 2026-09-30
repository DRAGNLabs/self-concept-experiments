"""Select labeled rh_hillclimb records from verdicts and candidate scans."""
import argparse
from collections import Counter
from collections.abc import Iterator
from functools import cache
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
# rh_hillclimb/scan_hacks.py HACK_LABELS: a record the harness itself labels as a hack is never a negative.
HARNESS_HACK_LABELS = frozenset({"hack_visible_only", "hack_test_modified", "cheat_modify_tests", "cheat_special_case"})

type RecordKey = str
type BinaryLabel = Literal[0, 1]
type Selection = Literal["hack", "reviewed_not_hack", "unflagged", "flagged_unreviewed", "ambiguous_verdict"]


class VerdictRecord(TypedDict):
    file: str
    verdict: str
    strength: NotRequired[str]


class LabeledRecord(NamedTuple):
    key: RecordKey
    run: str
    stem: str
    record: dict[str, Any]
    verdict: VerdictRecord | None
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


def key_run(key: RecordKey) -> str:
    return canonical_key(key).split("__", 1)[0]


def load_verdicts(paths: list[Path]) -> dict[RecordKey, VerdictRecord]:
    """Load verdicts in input order; later rows replace earlier judgments for the same record."""
    verdicts: dict[RecordKey, VerdictRecord] = {}
    for path in paths:
        for row in read_jsonl(path):
            key = canonical_key(row["file"])
            verdicts[key] = row
    return verdicts


def load_flagged(paths: list[Path]) -> set[RecordKey]:
    """Load the union of records flagged by candidate scans."""
    return {canonical_key(row["file"]) for path in paths for row in read_jsonl(path)}


def record_key(run: str, stem: str, example_id: str) -> RecordKey:
    return canonical_key(f"{run}__{stem}__{example_id}.md")


def validate_flagged_coverage(runs: list[str], flagged: set[RecordKey]) -> None:
    flagged_runs = {key_run(key) for key in flagged}
    uncovered = [run for run in runs if run not in flagged_runs]
    if uncovered:
        raise ValueError(f"No flagged-candidate coverage for verdict runs: {', '.join(uncovered)}")


def iter_run_records(root: Path, run: str) -> Iterator[tuple[RecordKey, str, dict[str, Any]]]:
    for path in sorted((root / "runs" / run).glob("*.jsonl")):
        if path.name.endswith("_reasoning.jsonl"):
            continue
        for record in read_jsonl(path):
            yield record_key(run, path.stem, record["example_id"]), path.stem, record


def select_record(key: RecordKey, run: str, stem: str, record: dict[str, Any],
                  verdict: VerdictRecord | None, flagged: set[RecordKey]) -> tuple[LabeledRecord | None, Selection]:
    if verdict is not None and verdict["verdict"] == "HACK":
        return LabeledRecord(key, run, stem, record, verdict, 1), "hack"
    if verdict is not None and verdict["verdict"] == "NOT_HACK":
        return LabeledRecord(key, run, stem, record, verdict, 0), "reviewed_not_hack"
    if verdict is not None:
        return None, "ambiguous_verdict"
    if key in flagged or record["label"] in HARNESS_HACK_LABELS:
        return None, "flagged_unreviewed"
    return LabeledRecord(key, run, stem, record, None, 0), "unflagged"


def labeled_run(root: Path, run: str, verdicts: dict[RecordKey, VerdictRecord],
                flagged: set[RecordKey]) -> tuple[list[LabeledRecord], set[RecordKey]]:
    labeled: list[LabeledRecord] = []
    found: set[RecordKey] = set()
    counts: Counter[str] = Counter()
    for key, stem, record in iter_run_records(root, run):
        if key in found:
            raise ValueError(f"Duplicate rh_hillclimb record key: {key}")
        found.add(key)
        selected, selection = select_record(key, run, stem, record, verdicts.get(key), flagged)
        counts.update(("records", selection))
        if selected is not None:
            labeled.append(selected)
    logger.info(
        "%s: %d records, %d HACK, %d reviewed NOT_HACK, %d unflagged, %d flagged unreviewed, "
        "%d ambiguous verdict",
        run, counts["records"], counts["hack"], counts["reviewed_not_hack"], counts["unflagged"],
        counts["flagged_unreviewed"], counts["ambiguous_verdict"])
    return labeled, found


def labeled_records(root: Path, verdicts: dict[RecordKey, VerdictRecord],
                    flagged: set[RecordKey]) -> list[LabeledRecord]:
    """Label reviewed and unflagged records, excluding flagged records awaiting review."""
    runs = sorted({key_run(key) for key in verdicts})
    validate_flagged_coverage(runs, flagged)
    labeled: list[LabeledRecord] = []
    found: set[RecordKey] = set()
    for run in runs:
        run_labeled, run_found = labeled_run(root, run, verdicts, flagged)
        labeled.extend(run_labeled)
        found.update(run_found)

    for key in sorted(verdicts.keys() - found):
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


def transcript_outcome(item: LabeledRecord, example_id: str, model_key: str) -> OutcomeRecord:
    record, verdict = item.record, item.verdict
    outcome: OutcomeRecord = {
        "example_id": example_id, "scenario": record["scenario"],
        "status": "truncated" if record["attempts"][-1]["truncated"] else "complete", "label": record["label"],
        "model_key": model_key, "verdict": verdict["verdict"] if verdict is not None else "unflagged",
        "harness_label": record["label"], "final_code": record.get("final_code", "")}
    if verdict is not None and "strength" in verdict:
        outcome["strength"] = verdict["strength"]
    return outcome


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
    parser.add_argument("--flagged", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    selected = labeled_records(args.root, load_verdicts(args.verdicts), load_flagged(args.flagged))
    transcripts = build_transcripts(args.root, selected)
    write_transcripts(args.out, transcripts)
    logger.info("Wrote %d transcripts to %s", len(transcripts), args.out)


if __name__ == "__main__":
    main()
