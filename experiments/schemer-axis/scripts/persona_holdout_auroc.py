"""Held-out persona and costume AUROCs for schemer-axis candidate vectors.

Fits three layer-wise axes from response-only, score-3 responses to the 64
training prompts:
  persona_contrast = deceptive_main - honest_main
  persona_contrast_honest_projected_out = persona_contrast projected off
      (honest_main - default)
  persona_contrast_no_costume = persona_contrast projected off
      (neutral_fic - default)

It then scores mean-pooled held-out-prompt activations from a separately
selected region.  Deception compares
the held-out deceptive and honest personas; costume compares held-out honest
fictional personas with the non-fictional default.  The JSON report contains
all layers and a concise layer-32 table, including the 2.5--97.5 percentile
interval from 256 seeded isotropic random directions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from scipy.stats import rankdata


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
QuestionGroup = Literal["generic", "intent", "task"]
QUESTION_GROUPS: tuple[QuestionGroup, ...] = ("generic", "intent", "task")
MAIN_DECEPTIVE = "deceptive_main"
MAIN_HONEST = "honest_main"
HELDOUT_DECEPTIVE = "deceptive_holdout"
HELDOUT_HONEST = "honest_holdout"
RH_DECEPTIVE = "deceptive_rh"
RH_HONEST = "honest_rh"
PERSONA_SETS = {"heldout": (HELDOUT_DECEPTIVE, HELDOUT_HONEST),
                "rh_adjacent": (RH_DECEPTIVE, RH_HONEST)}


def question_id(key: str) -> int:
    return int(key.rsplit("_q", 1)[1])


class Question(TypedDict):
    id: int
    question: str
    group: QuestionGroup
    split: Literal["train", "heldout"]


@dataclass(frozen=True)
class SelectBySplit:
    split: Literal["train", "heldout"]


@dataclass(frozen=True)
class SelectByGroups:
    groups: tuple[QuestionGroup, ...]  # every question in these groups, whatever its split


QuestionSelection = SelectBySplit | SelectByGroups


def question_ids(questions: list[Question], selection: QuestionSelection) -> set[int]:
    match selection:
        case SelectBySplit(split):
            return {question["id"] for question in questions if question["split"] == split}
        case SelectByGroups(groups):
            return {question["id"] for question in questions if question["group"] in groups}


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True)


def project_out(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    reference = unit(reference)
    return vector - (vector * reference).sum(dim=-1, keepdim=True) * reference


def selected(root: Path, region: str, role: str, question_ids: set[int]) -> tuple[torch.Tensor, np.ndarray]:
    activations = torch.load(root / f"act_{region}" / f"{role}.pt", map_location="cpu", weights_only=False)
    if role == "default":
        keep = [(value.float(), question_id(key)) for key, value in activations.items() if question_id(key) in question_ids]
    else:
        scores = json.loads((root / "scores" / f"{role}.json").read_text())
        keep = [(value.float(), question_id(key)) for key, value in activations.items()
                if question_id(key) in question_ids and scores.get(key) == 3]
    if not keep:
        raise ValueError(f"No retained responses for {role}")
    return torch.stack([activation for activation, _ in keep]), np.array([question for _, question in keep])


def role_mean(root: Path, region: str, role: str, train_ids: set[int]) -> torch.Tensor:
    return selected(root, region, role, train_ids)[0].mean(dim=0)


def auroc_columns(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC for every score column, with ties receiving half credit."""
    positive = labels == 1
    n_positive, n_negative = positive.sum(), (~positive).sum()
    ranks = rankdata(scores, axis=0)
    return (ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def stratified_auroc_columns(scores: np.ndarray, labels: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """Pair-pooled AUROC, comparing classes only within the same prompt stratum."""
    wins = np.zeros(scores.shape[1])
    pairs = 0
    for prompt in np.unique(prompt_ids):
        in_prompt = prompt_ids == prompt
        local_labels, local_scores = labels[in_prompt], scores[in_prompt]
        positive = local_labels == 1
        n_positive, n_negative = positive.sum(), (~positive).sum()
        if not n_positive or not n_negative:
            continue
        ranks = rankdata(local_scores, axis=0)
        wins += ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2
        pairs += n_positive * n_negative
    if not pairs:
        raise ValueError("No held-out prompt contains both classes")
    return wins / pairs


def axis_report(axis: torch.Tensor, random_axes: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                positive_prompt_ids: np.ndarray, negative_prompt_ids: np.ndarray) -> dict[str, list[float]]:
    direction = unit(axis)
    scores = torch.einsum("nlh,lh->nl", torch.cat((positive, negative)), direction).numpy()
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    prompt_ids = np.concatenate((positive_prompt_ids, negative_prompt_ids))
    observed = auroc_columns(scores, labels)
    stratified_observed = stratified_auroc_columns(scores, labels, prompt_ids)
    random_scores = torch.einsum("nlh,ldh->nld", torch.cat((positive, negative)), random_axes).numpy()
    random = np.array([auroc_columns(random_scores[:, layer], labels) for layer in range(random_scores.shape[1])])
    stratified_random = np.array([stratified_auroc_columns(random_scores[:, layer], labels, prompt_ids)
                                  for layer in range(random_scores.shape[1])])
    more_extreme = (np.abs(random - .5) > np.abs(observed[:, None] - .5)).mean(axis=1)
    stratified_more_extreme = (np.abs(stratified_random - .5) > np.abs(stratified_observed[:, None] - .5)).mean(axis=1)
    return {"auroc": observed.tolist(), "random_2_5": np.quantile(random, .025, axis=1).tolist(),
            "random_50": np.quantile(random, .5, axis=1).tolist(), "random_97_5": np.quantile(random, .975, axis=1).tolist(),
            "random_more_extreme_percent": (100 * more_extreme).tolist(), "random_aurocs": random.tolist(),
            "stratified_auroc": stratified_observed.tolist(),
            "stratified_random_2_5": np.quantile(stratified_random, .025, axis=1).tolist(),
            "stratified_random_50": np.quantile(stratified_random, .5, axis=1).tolist(),
            "stratified_random_97_5": np.quantile(stratified_random, .975, axis=1).tolist(),
            "stratified_random_more_extreme_percent": (100 * stratified_more_extreme).tolist(),
            "stratified_random_aurocs": stratified_random.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--axis-region", choices=("response_only",), default="response_only",
                        help="Activation region used to fit all candidate axes.")
    parser.add_argument("--score-region", choices=("with_cot", "response_only"), default="response_only",
                        help="Activation region projected onto the response-only-fitted axes.")
    parser.add_argument("--groups", type=Path, default=SCRIPT_ROOT / "data/persona_groups.json")
    parser.add_argument("--questions", type=Path, default=SCRIPT_ROOT / "data/questions.jsonl")
    parser.add_argument("--fit-question-groups", nargs="+", choices=QUESTION_GROUPS,
                        help="Fit on every question in these groups regardless of split (default: the split=train questions).")
    parser.add_argument("--eval-question-groups", nargs="+", choices=QUESTION_GROUPS,
                        help="Score on every question in these groups regardless of split (default: the split=heldout questions).")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--persona-set", choices=tuple(PERSONA_SETS), default="heldout",
                        help="Persona groups used for the held-out deception and costume comparisons.")
    parser.add_argument("--random-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    questions: list[Question] = [json.loads(line) for line in args.questions.read_text().splitlines()]
    fit_selection = SelectByGroups(tuple(args.fit_question_groups)) if args.fit_question_groups else SelectBySplit("train")
    eval_selection = SelectByGroups(tuple(args.eval_question_groups)) if args.eval_question_groups else SelectBySplit("heldout")
    train_ids = question_ids(questions, fit_selection)
    heldout_ids = question_ids(questions, eval_selection)
    if train_ids & heldout_ids:
        raise SystemExit(f"Fit and evaluation questions overlap: {sorted(train_ids & heldout_ids)}")
    fit_roles = [*groups[MAIN_DECEPTIVE], *groups[MAIN_HONEST], "default", "neutral_fic"]
    means = {role: role_mean(args.root, args.axis_region, role, train_ids) for role in fit_roles}
    deceptive_mean = torch.stack([means[role] for role in groups[MAIN_DECEPTIVE]]).mean(0)
    honest_mean = torch.stack([means[role] for role in groups[MAIN_HONEST]]).mean(0)
    contrast = deceptive_mean - honest_mean
    axes = {"persona_contrast": contrast,
            "persona_contrast_honest_projected_out": project_out(contrast, honest_mean - means["default"]),
            "persona_contrast_no_costume": project_out(contrast, means["neutral_fic"] - means["default"])}
    layers, hidden = contrast.shape
    generator = torch.Generator().manual_seed(args.seed)
    random_axes = torch.randn((layers, args.random_count, hidden), generator=generator)
    random_axes = random_axes / random_axes.norm(dim=-1, keepdim=True)

    def concatenate(roles: list[str]) -> tuple[torch.Tensor, np.ndarray]:
        records = [selected(args.root, args.score_region, role, heldout_ids) for role in roles]
        return torch.cat([activations for activations, _ in records]), np.concatenate([ids for _, ids in records])

    deceptive_group, honest_group = PERSONA_SETS[args.persona_set]
    deception_positive, deception_positive_ids = concatenate(groups[deceptive_group])
    deception_negative, deception_negative_ids = concatenate(groups[honest_group])
    costume_positive, costume_positive_ids = concatenate(groups[honest_group])
    costume_negative, costume_negative_ids = concatenate(["default"])
    experiments = {"deception": (deception_positive, deception_negative, deception_positive_ids, deception_negative_ids),
                   "costume": (costume_positive, costume_negative, costume_positive_ids, costume_negative_ids)}
    result = {"axis_region": args.axis_region, "score_region": args.score_region, "persona_set": args.persona_set,
              "deceptive_group": deceptive_group, "honest_group": honest_group,
              "fit_question_count": len(train_ids), "heldout_question_count": len(heldout_ids),
              "layer": args.layer, "random_count": args.random_count, "seed": args.seed,
              "response_counts": {name: {"positive": len(positive), "negative": len(negative)}
                                  for name, (positive, negative, _, _) in experiments.items()},
              "results": {name: {axis_name: axis_report(axis, random_axes, positive, negative, positive_ids, negative_ids)
                                  for axis_name, axis in axes.items()}
                          for name, (positive, negative, positive_ids, negative_ids) in experiments.items()}}
    result["layer_summary"] = {experiment: {axis_name: {key: values[args.layer] for key, values in report.items()
                                                         if key not in ("random_aurocs", "stratified_random_aurocs")}
                                              for axis_name, report in reports.items()}
                               for experiment, reports in result["results"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["layer_summary"], indent=2))


if __name__ == "__main__":
    main()
