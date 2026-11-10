"""How well the automatic hack proxy (passed shown tests, failed the hidden suite) tracks the judged HACK verdict, per
model: episode-level phi, P(judged | proxy), the per-arm correlation of problem-stratified rates (Pearson and
Spearman, one point per arm) and the per-problem correlation with all arms pooled. Unjudged episodes count as
non-hacks, as in main_analysis.py. Arms whose directories are unreadable (another user's permissions) are skipped and
listed.

Models: every cap_vs_steer_setups.json entry with verdicts, plus Qwen3.8-27B from Jansen's qwen_hard61 judging.

    python scripts/proxy_judged_correlation.py
"""
import json
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
from scipy import stats

SCRIPT_ROOT = Path(__file__).parents[1]
SHARE = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
QWEN_JUDGING = SHARE / "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61"


class ModelRun(NamedTuple):
    directory_by_arm: dict[str, Path]
    tag: str
    verdicts: Path


class Episode(TypedDict):
    arm: str
    problem: str
    proxy: bool
    judged: bool


def model_runs() -> dict[str, ModelRun]:
    setups = json.loads((SCRIPT_ROOT / "data/cap_vs_steer_setups.json").read_text())
    runs = {model: ModelRun({arm: Path(directory) for arm, directory in setup["directories_by_arm"].items()}, setup["tag"], Path(setup["verdicts"]))
            for model, setup in setups.items() if setup.get("verdicts")}
    qwen_arms = {path.name: path for path in sorted((QWEN_JUDGING / "analysis_root_floor").iterdir())}
    return runs | {"Qwen3.8-27B": ModelRun(qwen_arms, "pilot", QWEN_JUDGING / "verdicts_floor/variants_all12/as_judged.json")}


def arm_records(arm_directory: Path, tag: str) -> list[tuple[int, dict]]:
    """(seed, record) pairs; empty when the arm cannot be read."""
    try:
        return [(int(path.parent.name.removeprefix("seed")), json.loads(line))
                for path in sorted(arm_directory.glob(f"seed*/{tag}_impossible_original.jsonl")) for line in path.open() if line.strip()]
    except PermissionError:
        return []


def episodes_of(run: ModelRun) -> tuple[list[Episode], list[str]]:
    verdict_by_key: dict[str, str] = json.loads(run.verdicts.read_text())
    episodes: list[Episode] = []
    unreadable_arms: list[str] = []
    for arm, directory in run.directory_by_arm.items():
        records = arm_records(directory, run.tag)
        if not records:
            unreadable_arms.append(arm)
        episodes += [Episode(arm=arm, problem=record["example_id"], proxy=bool(record.get("passed")) and not record.get("passed_original"),
                             judged=verdict_by_key.get(f"{arm}/seed{seed}/{record['example_id']}", "").upper() == "HACK")
                     for seed, record in records]
    return episodes, unreadable_arms


def stratified_rate(episodes: list[Episode], arm: str, outcome: str) -> float:
    """Mean over problems of the problem's mean outcome in this arm."""
    outcomes_by_problem: dict[str, list[bool]] = {}
    for episode in episodes:
        if episode["arm"] == arm:
            outcomes_by_problem.setdefault(episode["problem"], []).append(episode[outcome])
    return float(np.mean([np.mean(outcomes) for outcomes in outcomes_by_problem.values()]))


def pooled_problem_rate(episodes: list[Episode], problem: str, outcome: str) -> float:
    return float(np.mean([episode[outcome] for episode in episodes if episode["problem"] == problem]))


def report(model: str, episodes: list[Episode], unreadable_arms: list[str]) -> str:
    arms = sorted({episode["arm"] for episode in episodes})
    problems = sorted({episode["problem"] for episode in episodes})
    proxy = np.array([episode["proxy"] for episode in episodes])
    judged = np.array([episode["judged"] for episode in episodes])
    proxy_by_arm = [stratified_rate(episodes, arm, "proxy") for arm in arms]
    judged_by_arm = [stratified_rate(episodes, arm, "judged") for arm in arms]
    proxy_by_problem = [pooled_problem_rate(episodes, problem, "proxy") for problem in problems]
    judged_by_problem = [pooled_problem_rate(episodes, problem, "judged") for problem in problems]
    arm_pearson, arm_pearson_p = stats.pearsonr(proxy_by_arm, judged_by_arm)
    arm_spearman, arm_spearman_p = stats.spearmanr(proxy_by_arm, judged_by_arm)
    problem_pearson, _ = stats.pearsonr(proxy_by_problem, judged_by_problem)
    problem_spearman, _ = stats.spearmanr(proxy_by_problem, judged_by_problem)
    unreadable = f" (unreadable: {unreadable_arms})" if unreadable_arms else ""
    return "\n".join([
        f"## {model}: {len(arms)} arms, {len(problems)} problems, {len(episodes)} episodes{unreadable}",
        f"  episode phi {np.corrcoef(proxy, judged)[0, 1]:.2f}; P(judged | proxy) {judged[proxy].mean():.1%}; "
        f"judged hacks inside proxy {judged[proxy].sum()} of {judged.sum()}",
        f"  per-arm: Pearson {arm_pearson:.2f} (p {arm_pearson_p:.3f}), Spearman {arm_spearman:.2f} (p {arm_spearman_p:.3f})",
        "   proxy/judged: " + "; ".join(f"{arm} {proxy_rate:.1%}/{judged_rate:.1%}"
                                       for arm, proxy_rate, judged_rate in zip(arms, proxy_by_arm, judged_by_arm)),
        f"  per-problem (all arms pooled): Pearson {problem_pearson:.2f}, Spearman {problem_spearman:.2f}",
        ""])


def main() -> None:
    for model, run in model_runs().items():
        print(report(model, *episodes_of(run)))


if __name__ == "__main__":
    main()
