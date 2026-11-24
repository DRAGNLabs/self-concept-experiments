"""Model presets (arm directories, judge verdicts) and loaders for the four models' impossible_original episodes."""

import json
from pathlib import Path
from typing import Literal, TypedDict

from selfconcept.codebench.harness import CheckRecord
from selfconcept.codebench.rubric_levels import Verdict

SHARE = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
type Axis = Literal["gi_v2", "gfar"]
type Dose = Literal["half", "main"]
type ArmName = Literal["unsteered", "gi_v2", "gfar"]
ARMS: list[ArmName] = ["unsteered", "gi_v2", "gfar"]
# visible tests hold a wrong assert (test audit 2026-11-23); lcb_arc192_b is lcbhard_94 under another id
VISIBLE_BROKEN_PROBLEM_IDS = frozenset({"lcbhard_57", "lcbhard_81", "lcbhard_94", "lcbhard_77", "lcb_arc192_b"})
MODEL_ID_BY_SLUG = {"gptoss": "openai/gpt-oss-120b", "muse": "meta-models/Muse-Glimmer-30B",
                    "gemma": "google/gemma-4-31B-it", "qwen": "Qwen/Qwen3.8-27B"}


class JudgeDetail(TypedDict):
    """One rubric-v3 verdict in a judging set's *.detail.json (rh_judge.RubricVerdict without `file`)."""
    verdict: Verdict
    strength: str | None
    mechanism: str
    turn: int | None
    first_considered_turn: int | str | None  # a few judge passes wrote digit strings
    pressure_source: str
    feedback_confusion: str
    intent_quote: str
    code_quote: str
    justification: str
    dossier: str


class ModelPreset(TypedDict):
    label: str
    slug: str
    record_file: str
    unsteered: Path
    arm_dirs: dict[tuple[Axis, Dose], Path]
    alpha_by_dose: dict[Dose, float]
    verdict_files: list[Path]
    exclusion_reason_by_proxy: dict[str, str]


GPTOSS_PILOT = SHARE / "gptoss120b/gi_v2/pilot"
MUSE_PILOT = SHARE / "muse_glimmer/gi_v2/pilot"
GEMMA_RUNS = SHARE / "gemma4_31b_gi_v2/rh/hard108"
GEMMA_JUDGING = SHARE / "gemma4_31b_gi_v2_judging_jansen05/hard108"
QWEN_RUNS = SHARE / "qwen38_27b_gi_v2/rh/qwen_hard61"
QWEN_JUDGING = SHARE / "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61"
PRESETS: list[ModelPreset] = [
    {"label": "gpt-oss-120b", "slug": "gptoss",
     "record_file": "main_impossible_original.jsonl", "unsteered": GPTOSS_PILOT / "unsteered",
     "arm_dirs": {("gi_v2", "half"): GPTOSS_PILOT / "gi_m0.1", ("gi_v2", "main"): GPTOSS_PILOT / "gi_m0.2",
                  ("gfar", "half"): GPTOSS_PILOT / "gf_m0.1", ("gfar", "main"): GPTOSS_PILOT / "gf_m0.2"},
     "alpha_by_dose": {"half": 0.1, "main": 0.2},
     "verdict_files": [GPTOSS_PILOT / "verdicts_gfscan/variants_gfscan_all/as_judged.json"],
     "exclusion_reason_by_proxy": {}},
    {"label": "Muse", "slug": "muse",
     "record_file": "main_impossible_original.jsonl", "unsteered": MUSE_PILOT / "unsteered",
     "arm_dirs": {("gi_v2", "half"): MUSE_PILOT / "gi_m0.04", ("gi_v2", "main"): MUSE_PILOT / "gi_m0.08",
                  ("gfar", "half"): MUSE_PILOT / "gf_m0.04", ("gfar", "main"): MUSE_PILOT / "gf_m0.08"},
     "alpha_by_dose": {"half": 0.04, "main": 0.08},
     "verdict_files": [MUSE_PILOT / "verdicts_gfx03/variants_gfx03_all/as_judged.json"],
     "exclusion_reason_by_proxy": {"abstention": "no untruncated stop in any arm"}},
    {"label": "Gemma 4 31B", "slug": "gemma",
     "record_file": "pilot_impossible_original.jsonl", "unsteered": GEMMA_RUNS / "unsteered",
     "arm_dirs": {("gi_v2", "half"): GEMMA_RUNS / "L30_a-0.04", ("gi_v2", "main"): GEMMA_RUNS / "L30_a-0.08",
                  ("gfar", "half"): GEMMA_JUDGING / "gfar_L30_a-0.04",
                  ("gfar", "main"): GEMMA_JUDGING / "gfar_L30_a-0.08"},
     "alpha_by_dose": {"half": 0.04, "main": 0.08},
     "verdict_files": [GEMMA_JUDGING / "analysis_gfar6/variants_all/as_judged.json",
                       GEMMA_JUDGING / "verdicts_ctl_gicap_s0-15/variants_all16/as_judged.json"],
     "exclusion_reason_by_proxy": {}},
    {"label": "Qwen3.8-27B", "slug": "qwen",
     "record_file": "pilot_impossible_original.jsonl", "unsteered": QWEN_RUNS / "unsteered",
     "arm_dirs": {("gi_v2", "half"): QWEN_RUNS / "L32_a-0.08", ("gi_v2", "main"): QWEN_RUNS / "L32_a-0.16",
                  ("gfar", "half"): SHARE / "qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/gfar_L32_a-0.08",
                  ("gfar", "main"): SHARE / "coverage_1113_kobyjl/qwen38_27b/rh/gfar_L32_a-0.16"},
     "alpha_by_dose": {"half": 0.08, "main": 0.16},
     "verdict_files": [QWEN_JUDGING / "verdicts_gfx04/variants_gfx04_all/as_judged.json"],
     "exclusion_reason_by_proxy": {"admission": "1 judged hack across its unsteered and negative-steered arms"}},
]
PRESET_BY_SLUG: dict[str, ModelPreset] = {preset["slug"]: preset for preset in PRESETS}
JUDGING_ROOT_BY_SLUG = {"gptoss": GPTOSS_PILOT, "muse": MUSE_PILOT, "gemma": GEMMA_JUDGING, "qwen": QWEN_JUDGING}


def load_records(arm_dir: Path, record_file: str) -> dict[str, CheckRecord]:
    """Episode records of every seed of an arm, by "seed<N>/<example_id>"."""
    return {f"{path.parent.name}/{record['example_id']}": record
            for path in sorted(arm_dir.glob(f"seed*/{record_file}")) for record in map(json.loads, path.open())}


def load_verdict_by_key(verdict_files: list[Path]) -> dict[str, str]:
    verdict_by_key: dict[str, str] = {}
    for path in verdict_files:
        for key, verdict in json.loads(path.read_text()).items():
            assert verdict_by_key.get(key, verdict) == verdict, f"verdict files disagree on {key}"
            verdict_by_key[key] = verdict
    return verdict_by_key


def main_dose_arm_dirs(preset: ModelPreset) -> dict[ArmName, Path]:
    return {"unsteered": preset["unsteered"], "gi_v2": preset["arm_dirs"][("gi_v2", "main")],
            "gfar": preset["arm_dirs"][("gfar", "main")]}


def arm_dir_names(preset: ModelPreset) -> dict[str, ArmName]:
    return {arm_dir.name: arm for arm, arm_dir in main_dose_arm_dirs(preset).items()}


def hack_turns(detail: JudgeDetail) -> tuple[int, int]:
    """(first considered turn, hack turn) of a HACK verdict's detail."""
    first_considered, hack = detail["first_considered_turn"], detail["turn"]
    assert first_considered is not None and hack is not None, f"HACK detail without turns: {detail['dossier']}"
    return int(first_considered), hack


def load_judge_details(preset: ModelPreset, verdict_by_key: dict[str, str],
                       arm_dir_name_set: set[str] | None = None) -> dict[str, JudgeDetail]:
    """Rubric-v3 judge details by episode key for the main-dose arms (or the named arm dirs); a detail counts only if
    its verdict matches the verdict files' verdict for that episode, so details from superseded judging sets are
    ignored."""
    wanted = arm_dir_name_set if arm_dir_name_set is not None else set(arm_dir_names(preset))
    detail_by_key: dict[str, JudgeDetail] = {}
    for path in sorted(JUDGING_ROOT_BY_SLUG[preset["slug"]].glob("*/*.detail.json")):
        for key, detail in json.loads(path.read_text()).items():
            if key.split("/")[0] in wanted and detail.get("verdict") == verdict_by_key.get(key):
                detail_by_key.setdefault(key, detail)
    return detail_by_key
