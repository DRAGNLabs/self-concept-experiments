"""Per-(target, other) recognition statistics and the mechanical GO/NO-GO yield check per target.

Writes run_dir/judgements/summary.json.

Usage (from experiments/self-recognition):
    python scripts/2_summarize.py --run.n_train_prompts 189 --run.target_samples 4 --run.min_robust 300
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from selfconcept.self_recognition.judging import read_transcripts_by_model
from selfconcept.self_recognition.schemas import DEFAULT_RUN_DIR, Split, prompts_path, read_judgements, read_prompts
from selfconcept.self_recognition.success_filter import (
    POOLED_OTHER_MODEL,
    GoDecision,
    PairSummary,
    go_decision,
    name_leaks_inflate_accuracy,
    summarize_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    run_dir: Path = DEFAULT_RUN_DIR
    data_dir: Path = Path("data")
    split: Split | None = "train"
    n_train_prompts: int = 189
    target_samples: int = 4
    min_robust: int = 300


class TargetSummary(TypedDict):
    decision: GoDecision
    exclude_name_leaks: bool
    pairs: list[PairSummary]


def summary_path(run_dir: Path) -> Path:
    return run_dir / "judgements" / "summary.json"


def format_optional(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def pair_summary_line(summary: PairSummary) -> str:
    accuracy = summary["accuracy"]
    accuracy_text = "-" if accuracy is None else f"{accuracy['accuracy']:.3f} [{accuracy['ci_low']:.3f}, {accuracy['ci_high']:.3f}]"
    return (
        f"{summary['other_model']:<28} items={summary['items']:<5} skipped={summary['skipped_truncated_or_empty']:<4} "
        f"acc={accuracy_text} debiased={format_optional(summary['debiased_accuracy'])} "
        f"pos_bias={summary['position_bias_rate']:.3f} p1={format_optional(summary['p_choose_1'])} "
        f"unparseable={summary['unparseable_rate']:.3f} robust={summary['robust_count']} ({summary['robust_rate']:.3f}) "
        f"null={format_optional(summary['null_robust_rate'])} purity={format_optional(summary['purity'])} "
        f"longer={summary['longer_text_baseline']:.3f} name_leak={summary['name_leak_rate']:.3f}"
    )


def length_selection_line(summary: PairSummary) -> str:
    length = summary["length_selection"]
    matched = length["length_matched_accuracy"]
    matched_text = "-" if matched is None else f"{matched['accuracy']:.3f} [{matched['ci_low']:.3f}, {matched['ci_high']:.3f}]"
    return (
        f"{summary['other_model']:<28} spearman(len, chose own)={length['selection_spearman']:.3f} "
        f"[{length['selection_spearman_ci_low']:.3f}, {length['selection_spearman_ci_high']:.3f}] "
        f"spearman(len, P(own))={length['own_probability_spearman']:.3f} own_longer={length['own_longer_rate']:.3f} "
        f"own_longer|recognised={length['own_longer_rate_among_recognised']:.3f} "
        f"within_20%={length['length_matched_item_rate']:.3f} acc_within_20%={matched_text}"
    )


def decision_line(decision: GoDecision, exclude_name_leaks: bool) -> str:
    verdict = decision["verdict"]
    if decision["failed_criteria"]:
        verdict += f" (failed: {', '.join(decision['failed_criteria'])}; K needed for min_robust: {decision['samples_needed_for_min_robust']})"
    return (
        f"{decision['target_model']}: observed recognised={decision['observed_robust_count']} "
        f"projected={decision['projected_robust_count']:.0f} acc_ci_low={format_optional(decision['accuracy_ci_low'])} "
        f"purity={format_optional(decision['purity'])} -> {verdict}; "
        f"name leaks {'EXCLUDED' if exclude_name_leaks else 'kept'} from recognised pairs"
    )


def main(run: RunConfig) -> None:
    logger.info("%s", run)
    prompt_ids = {prompt["id"] for prompt in read_prompts(prompts_path(run.data_dir)) if run.split is None or prompt["split"] == run.split}
    judgements = [
        judgement
        for path in sorted((run.run_dir / "judgements").glob("*.jsonl"))
        for judgement in read_judgements(path)
        if judgement["prompt_id"] in prompt_ids
    ]
    summaries = summarize_pairs(judgements, read_transcripts_by_model(run.run_dir))
    target_summary_by_model: dict[str, TargetSummary] = {}
    for target_model in sorted({summary["target_model"] for summary in summaries}):
        target_rows = [summary for summary in summaries if summary["target_model"] == target_model]
        pooled = next(summary for summary in target_rows if summary["other_model"] == POOLED_OTHER_MODEL)
        target_summary_by_model[target_model] = {
            "decision": go_decision(target_rows, run.n_train_prompts, run.target_samples, run.min_robust),
            "exclude_name_leaks": name_leaks_inflate_accuracy(pooled),
            "pairs": target_rows,
        }

    for target_model, target_summary in target_summary_by_model.items():
        logger.info("Target %s\n%s", target_model, "\n".join(pair_summary_line(row) for row in target_summary["pairs"]))
    for target_model, target_summary in target_summary_by_model.items():
        logger.info("Length vs selection, target %s\n%s", target_model, "\n".join(length_selection_line(row) for row in target_summary["pairs"]))
    for target_summary in target_summary_by_model.values():
        logger.info("%s", decision_line(target_summary["decision"], target_summary["exclude_name_leaks"]))

    output_path = summary_path(run.run_dir)
    output_path.write_text(json.dumps(target_summary_by_model, indent=2))
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
