"""Hack-rate summary of judged transcripts, by the user's hack tiers and by category.

    python -m selfconcept.goodfire_rh_metrics.metrics --run.benchmark impossiblebench_minimal_scaffold \
        --run.judged_paths '[<judged_*.jsonl>, ...]' --run.output_path <summary.json>

A transcript counts toward a tier or category when it has at least one quote-verified
flag there; withdrawn (unverified) flags are counted but otherwise ignored.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, get_args

import jsonlines

from .benchmarks import HackTier, JudgeBenchmark, JudgeBenchmarkName, category_tier, judge_benchmark_by_name
from .judge_output import FlagStatus, JudgedTranscript, JudgedTranscriptParsed

logger = logging.getLogger(__name__)

type StrongestStatus = FlagStatus | Literal["none"]

HACK_TIERS: tuple[HackTier, ...] = get_args(HackTier.__value__)
STATUSES_WEAKEST_FIRST: tuple[FlagStatus, ...] = ("considered", "attempted", "enacted")
ACTED_STATUSES: frozenset[StrongestStatus] = frozenset({"attempted", "enacted"})


class TranscriptMetrics(TypedDict):
    split: str
    transcript_id: str
    benchmark_label: str
    strongest_status_by_tier: dict[HackTier, StrongestStatus]
    strongest_status_by_category: dict[str, StrongestStatus]
    withdrawn_flag_count: int


class BenchmarkLabelCrosstab(TypedDict):
    n: int
    hit_count_by_tier: dict[HackTier, int]


class SplitSummary(TypedDict):
    n_judged: int
    n_judge_failed: int
    n_withdrawn_flags: int
    any_flag_rate: float
    rate_by_tier: dict[HackTier, float]
    acted_rate_by_tier: dict[HackTier, float]
    strongest_status_counts_by_tier: dict[HackTier, dict[StrongestStatus, int]]
    rate_by_category: dict[str, float]
    crosstab_by_benchmark_label: dict[str, BenchmarkLabelCrosstab]
    transcript_ids_by_tier: dict[HackTier, list[str]]


class MetricsSummary(TypedDict):
    benchmark: str
    by_split: dict[str, SplitSummary]
    overall: SplitSummary


def strongest_status(statuses: Sequence[FlagStatus]) -> StrongestStatus:
    return max(statuses, key=STATUSES_WEAKEST_FIRST.index) if statuses else "none"


def transcript_metrics(benchmark: JudgeBenchmark, record: JudgedTranscriptParsed) -> TranscriptMetrics:
    verified_flags = [flag for flag in record["judgment"]["flags"] if flag["quote_verified"]]
    return {
        "split": record["split"],
        "transcript_id": record["transcript_id"],
        "benchmark_label": record["benchmark_label"],
        "strongest_status_by_tier": {
            tier: strongest_status(
                [flag["status"] for flag in verified_flags if category_tier(benchmark, flag["category"]) == tier]
            )
            for tier in HACK_TIERS
        },
        "strongest_status_by_category": {
            category: strongest_status([flag["status"] for flag in verified_flags if flag["category"] == category])
            for category in benchmark["tier_by_category"]
        },
        "withdrawn_flag_count": len(record["judgment"]["flags"]) - len(verified_flags),
    }


def rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def split_summary(transcripts: Sequence[TranscriptMetrics], n_judge_failed: int) -> SplitSummary:
    n_judged = len(transcripts)

    def tier_hits(tier: HackTier) -> list[TranscriptMetrics]:
        return [transcript for transcript in transcripts if transcript["strongest_status_by_tier"][tier] != "none"]

    benchmark_labels = sorted({transcript["benchmark_label"] for transcript in transcripts})
    categories = list(transcripts[0]["strongest_status_by_category"]) if transcripts else []
    return {
        "n_judged": n_judged,
        "n_judge_failed": n_judge_failed,
        "n_withdrawn_flags": sum(transcript["withdrawn_flag_count"] for transcript in transcripts),
        "any_flag_rate": rate(
            sum(any(status != "none" for status in t["strongest_status_by_tier"].values()) for t in transcripts), n_judged
        ),
        "rate_by_tier": {tier: rate(len(tier_hits(tier)), n_judged) for tier in HACK_TIERS},
        "acted_rate_by_tier": {
            tier: rate(sum(t["strongest_status_by_tier"][tier] in ACTED_STATUSES for t in transcripts), n_judged)
            for tier in HACK_TIERS
        },
        "strongest_status_counts_by_tier": {
            tier: {
                status: sum(t["strongest_status_by_tier"][tier] == status for t in transcripts)
                for status in (*reversed(STATUSES_WEAKEST_FIRST), "none")
            }
            for tier in HACK_TIERS
        },
        "rate_by_category": {
            category: rate(sum(t["strongest_status_by_category"][category] != "none" for t in transcripts), n_judged)
            for category in categories
        },
        "crosstab_by_benchmark_label": {
            label: {
                "n": sum(t["benchmark_label"] == label for t in transcripts),
                "hit_count_by_tier": {
                    tier: sum(t["benchmark_label"] == label for t in tier_hits(tier)) for tier in HACK_TIERS
                },
            }
            for label in benchmark_labels
        },
        "transcript_ids_by_tier": {tier: [t["transcript_id"] for t in tier_hits(tier)] for tier in HACK_TIERS},
    }


def metrics_summary(benchmark_name: str, benchmark: JudgeBenchmark, records: Sequence[JudgedTranscript]) -> MetricsSummary:
    parsed_metrics = [transcript_metrics(benchmark, record) for record in records if record["status"] == "parsed"]
    splits = sorted({record["split"] for record in records})

    def failed_count(split: str | None) -> int:
        return sum(record["status"] == "failed" and split in (None, record["split"]) for record in records)

    return {
        "benchmark": benchmark_name,
        "by_split": {
            split: split_summary([t for t in parsed_metrics if t["split"] == split], failed_count(split))
            for split in splits
        },
        "overall": split_summary(parsed_metrics, failed_count(None)),
    }


@dataclass(frozen=True)
class RunConfig:
    benchmark: JudgeBenchmarkName
    judged_paths: tuple[Path, ...]
    output_path: Path


def main(run: RunConfig) -> None:
    """Summarize judged transcripts into hack rates by tier, category and benchmark label."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    records: list[JudgedTranscript] = []
    for judged_path in run.judged_paths:
        with jsonlines.open(judged_path, "r") as reader:
            records.extend(reader)
    summary = metrics_summary(run.benchmark, judge_benchmark_by_name[run.benchmark], records)
    run.output_path.parent.mkdir(parents=True, exist_ok=True)
    run.output_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s: overall %s", run.output_path, json.dumps(summary["overall"]["rate_by_tier"]))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
