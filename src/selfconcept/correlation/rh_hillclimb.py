"""Select labeled rh_hillclimb records from verdicts and candidate scans."""
from collections import Counter
from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

from selfconcept.common.jsonl import read_jsonl

logger = logging.getLogger(__name__)

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
    if key in flagged:
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
