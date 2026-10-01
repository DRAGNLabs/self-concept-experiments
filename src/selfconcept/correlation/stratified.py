"""Stratified AUROC: hacks are compared only with non-hacks from the same base problem.

usage: python -m selfconcept.correlation.stratified RUN_DIR [RUN_DIR ...] --out RESULT_DIR
Run directories are shards of one transcript scoring run (same measurement and primary layer).
"""
import argparse
from collections import defaultdict
from collections.abc import Sequence
import json
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
from scipy.stats import rankdata

from selfconcept.common.jsonl import read_jsonl
from .analyze import (
    binary_outcome, BinaryOutcome, earliest_turns, load_generations, PRIMARY_REGION, REGIONS, resolve_primary_layer)
from .records import GenerationRecord, Manifest, OutcomeRecord, Region

ALL_SCENARIOS = "all"


class StratifiedExample(NamedTuple):
    score: float
    outcome: BinaryOutcome
    stratum: str


class StratifiedAuroc(NamedTuple):
    auroc: float | None
    n_pairs: int


class ConfidenceInterval(NamedTuple):
    low: float
    high: float


def mann_whitney_u(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Pairs with the positive score higher, ties counting 1/2: positive rank sum minus its minimum possible value."""
    positive_ranks = rankdata(positive_scores + negative_scores)[:len(positive_scores)]
    return float(positive_ranks.sum()) - len(positive_scores) * (len(positive_scores) + 1) / 2


def stratified_auroc(examples: Sequence[StratifiedExample]) -> StratifiedAuroc:
    """Within-stratum (hack, non-hack) pairs pooled over strata; a higher hack score counts 1, a tie 1/2."""
    scores_by_stratum_and_outcome: defaultdict[tuple[str, BinaryOutcome], list[float]] = defaultdict(list)
    for example in examples:
        scores_by_stratum_and_outcome[example.stratum, example.outcome].append(example.score)
    concordant_pairs, n_pairs = 0., 0
    for stratum in {example.stratum for example in examples}:
        positive_scores = scores_by_stratum_and_outcome[stratum, 1]
        negative_scores = scores_by_stratum_and_outcome[stratum, 0]
        if not positive_scores or not negative_scores:
            continue
        concordant_pairs += mann_whitney_u(positive_scores, negative_scores)
        n_pairs += len(positive_scores) * len(negative_scores)
    return StratifiedAuroc(concordant_pairs / n_pairs if n_pairs else None, n_pairs)


def stratum_bootstrap_interval(examples: Sequence[StratifiedExample], samples: int = 1000,
                               seed: int = 1729) -> ConfidenceInterval | None:
    """95% interval from resampling whole strata; a stratum drawn twice enters as two separate strata."""
    examples_by_stratum: defaultdict[str, list[StratifiedExample]] = defaultdict(list)
    for example in examples:
        examples_by_stratum[example.stratum].append(example)
    strata = sorted(examples_by_stratum)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(samples):
        drawn_strata = [strata[index] for index in rng.integers(len(strata), size=len(strata))]
        resampled = [example._replace(stratum=f"{draw}:{stratum}")
                     for draw, stratum in enumerate(drawn_strata) for example in examples_by_stratum[stratum]]
        if (auroc := stratified_auroc(resampled).auroc) is not None:
            draws.append(auroc)
    if not draws:
        return None
    low, high = np.quantile(draws, [.025, .975]).tolist()
    return ConfidenceInterval(low, high)


class ScoredOutcome(NamedTuple):
    outcome: OutcomeRecord
    scored_turn: GenerationRecord


class StratifiedCell(TypedDict):
    scenario: str
    layer: int
    region: Region
    primary: bool
    n_strata: int
    n_positive: int
    n_negative: int
    n_pairs: int
    auroc: float | None
    ci: ConfidenceInterval | None


class StratifiedAnalysis(TypedDict):
    measurement: str
    primary_layer: int
    directories: list[str]
    cells: list[StratifiedCell]


def measurement_and_primary_layer(directory: Path) -> tuple[str, int]:
    manifest: Manifest = json.loads((directory / "manifest.json").read_text())
    measurement_name = manifest["args"].get("measurement", "cotness")
    assert isinstance(measurement_name, str)
    primary_layer = resolve_primary_layer(directory, manifest, measurement_name)
    if primary_layer is None:
        raise ValueError(f"{directory} has no usable measurement")
    return measurement_name, primary_layer


def scored_outcomes(directory: Path, measurement_name: str) -> dict[tuple[str, str], ScoredOutcome]:
    scored_turn_by_key = earliest_turns(load_generations(directory, measurement_name))
    outcomes: list[OutcomeRecord] = read_jsonl(directory / "outcomes.jsonl")
    return {(outcome["scenario"], outcome["example_id"]): ScoredOutcome(outcome, scored_turn_by_key[key])
            for outcome in outcomes if (key := (outcome["scenario"], outcome["example_id"])) in scored_turn_by_key}


def merged_scored_outcomes(directories: list[Path], measurement_name: str) -> list[ScoredOutcome]:
    merged: dict[tuple[str, str], ScoredOutcome] = {}
    for directory in directories:
        shard = scored_outcomes(directory, measurement_name)
        if duplicates := merged.keys() & shard.keys():
            raise ValueError(f"{directory} repeats {len(duplicates)} examples, e.g. {min(duplicates)}")
        merged |= shard
    return list(merged.values())


def examples_by_cell(scored: list[ScoredOutcome]) -> dict[tuple[str, str, Region], list[StratifiedExample]]:
    """Every example enters its own scenario's cells and the pooled cells; problems never repeat across scenarios."""
    cell_examples: defaultdict[tuple[str, str, Region], list[StratifiedExample]] = defaultdict(list)
    for record, scored_turn in scored:
        outcome, _ = binary_outcome(record)
        if outcome is None:
            continue
        if "stratum" not in record:
            raise ValueError(f"Outcome {record['example_id']} has no stratum; build transcripts with --stratified")
        for layer, summaries_by_region in scored_turn["scores"].items():
            for region in REGIONS:
                if region in summaries_by_region and (score := summaries_by_region[region]["mean"]) is not None:
                    example = StratifiedExample(score, outcome, record["stratum"])
                    cell_examples[record["scenario"], layer, region].append(example)
                    cell_examples[ALL_SCENARIOS, layer, region].append(example)
    return cell_examples


def stratified_cell(scenario: str, layer: str, region: Region, examples: list[StratifiedExample], primary_layer: int,
                    bootstrap_samples: int) -> StratifiedCell:
    outcomes_by_stratum: defaultdict[str, set[BinaryOutcome]] = defaultdict(set)
    for example in examples:
        outcomes_by_stratum[example.stratum].add(example.outcome)
    auroc, n_pairs = stratified_auroc(examples)
    return {"scenario": scenario, "layer": int(layer), "region": region,
            "primary": int(layer) == primary_layer and region == PRIMARY_REGION,
            "n_strata": sum(outcomes == {0, 1} for outcomes in outcomes_by_stratum.values()),
            "n_positive": sum(example.outcome for example in examples),
            "n_negative": sum(not example.outcome for example in examples),
            "n_pairs": n_pairs, "auroc": auroc, "ci": stratum_bootstrap_interval(examples, bootstrap_samples)}


def analyze_stratified(directories: list[Path], bootstrap_samples: int = 1000) -> StratifiedAnalysis:
    measurement_and_layers = {measurement_and_primary_layer(directory) for directory in directories}
    if len(measurement_and_layers) != 1:
        raise ValueError(f"Shards disagree on measurement or primary layer: {measurement_and_layers}")
    [(measurement_name, primary_layer)] = measurement_and_layers
    cell_examples = examples_by_cell(merged_scored_outcomes(directories, measurement_name))
    return {"measurement": measurement_name, "primary_layer": primary_layer,
            "directories": [str(directory) for directory in directories],
            "cells": [stratified_cell(scenario, layer, region, examples, primary_layer, bootstrap_samples)
                      for (scenario, layer, region), examples in sorted(cell_examples.items())]}


def render_markdown(result: StratifiedAnalysis) -> str:
    lines = [f"# Stratified {result['measurement']} AUROC", "",
             f"Primary: scored-turn {PRIMARY_REGION} content at layer {result['primary_layer']}. Hacks are compared only "
             "with non-hacks of the same base problem; the 95% CI resamples whole problems.", "",
             "| Scenario | strata | hacks | non-hacks | pairs | AUROC | 95% CI |", "|---|---:|---:|---:|---:|---:|---|"]
    for cell in result["cells"]:
        if cell["primary"]:
            ci = f"{cell['ci'].low:.3f}–{cell['ci'].high:.3f}" if cell["ci"] else "—"
            auroc = f"{cell['auroc']:.3f}" if cell["auroc"] is not None else "—"
            lines.append(f"| {cell['scenario']} | {cell['n_strata']} | {cell['n_positive']} | {cell['n_negative']} "
                         f"| {cell['n_pairs']} | {auroc} | {ci} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = analyze_stratified(args.directories, args.bootstrap)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stratified_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.out / "stratified_analysis.md").write_text(render_markdown(result))


if __name__ == "__main__":
    main()
