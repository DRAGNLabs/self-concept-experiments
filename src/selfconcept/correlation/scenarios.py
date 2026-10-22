"""Run benchmark scenarios through measured generation and record their outcomes."""
from collections.abc import Callable, Mapping
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, cast, NotRequired, TypedDict

from selfconcept.codebench import harness
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.common.jsonl import append_jsonl, read_jsonl
from selfconcept.common.paths import REPO_ROOT
from selfconcept.soo.evaluate import build_prompt, classify, SUFFIXES
from .config import RunConfig
from .generate import GenerationSettings, measured_generate, MeasuredModel, sampling_kwargs
from .records import GenerationRecord, OutcomeRecord

logger = logging.getLogger(__name__)

CODE_SCENARIOS = frozenset({"evilgenie", "impossible_original", "impossible_oneoff", "impossible_conflicting"})
APOLLO_SCENARIOS = frozenset({"roleplaying", "insider_trading", "sandbagging"})

type SandbaggingGrader = Callable[[str], dict[str, Any]]


class DeceptionExample(TypedDict):
    example_id: str
    prompt: NotRequired[str]
    messages: NotRequired[Conversation]
    answer_prefix: NotRequired[str]
    honest_answer: NotRequired[str]
    deceptive_answer: NotRequired[str]
    expected: NotRequired[str]
    topic: NotRequired[str]
    correct_answer: NotRequired[str]


def load_soo_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"experiments/soo/scripts/{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiments/soo/scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_data_path(scenario: str) -> Path:
    if scenario in CODE_SCENARIOS:
        return REPO_ROOT / f"benchmarks/codebench/data/{scenario}.jsonl"
    name = scenario.removesuffix("_mirrored")
    folder = "eval_mirrored" if scenario.endswith("_mirrored") else "eval_apollo" if name in APOLLO_SCENARIOS else "eval"
    return REPO_ROOT / f"experiments/soo/data/{folder}/{name}.jsonl"


def select_examples[ExampleT: Mapping[str, object]](rows: list[ExampleT], count: int, scenario: str,
                                                     offset: int = 0) -> list[ExampleT]:
    if count < 1 or offset < 0:
        raise ValueError("Positive count and nonnegative offset required")
    groups = ([[row for row in rows if row["expected"] == half] for half in ("sandbag", "normal")]
              if scenario == "sandbagging" else [rows])
    if any(len(group) < offset + count for group in groups):
        raise ValueError(f"Insufficient examples in {scenario} for offset={offset}, n={count}")
    return [row for group in groups for row in group[offset:offset + count]]


def run_code_example(measured_model: MeasuredModel, settings: GenerationSettings, scenario: str,
                     example: dict[str, Any], max_attempts: int, model_key: str) -> OutcomeRecord:
    generation_records: list[GenerationRecord] = []

    def generate_turn(messages: list[dict[str, Any]], turn: int) -> tuple[str, bool]:
        record = measured_generate(measured_model, settings, scenario, example["example_id"],
                                   cast(Conversation, messages), turn)
        generation_records.append(record)
        # A partial final answer/code is never a completed behavioral submission.
        return (record["response"] if record["status"] == "complete" else ""), record["truncated"]

    run_harness = harness.run_workspace_example if scenario == "evilgenie" else harness.run_check_example
    harness_record = run_harness(cast(Any, example), generate_turn, max_attempts, state_attempt_budget=False)
    status = generation_records[-1]["status"]
    return cast(OutcomeRecord, harness_record | {
        "status": status, "behavior_label": harness_record["label"] if status == "complete" else None,
        "generation_turns": len(generation_records), "model_key": model_key})


def deception_messages(example: DeceptionExample, name: str, suffix: str) -> Conversation:
    messages: Conversation = ([{**turn} for turn in example["messages"]] if "messages" in example else [
        {"role": "user", "content": build_prompt(dict(example), SUFFIXES[suffix] if name not in ("roleplaying", "sandbagging") else None, False)}])
    if "answer_prefix" not in example or not example["answer_prefix"]:
        return messages
    instruction = f"\n\nBegin your final answer with {example['answer_prefix']!r}."
    return [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + instruction}]


def sandbagging_fields(example: DeceptionExample, response: str,
                       grade_sandbagging: SandbaggingGrader) -> tuple[str, dict[str, Any]]:
    evidence = grade_sandbagging(response)
    label = "deceptive" if evidence["disagree"] is True else "honest" if evidence["disagree"] is False else "unparsed"
    return label, {"evidence": evidence, "expected": example.get("expected"), "topic": example.get("topic"),
                   "correct": evidence["final_answer"] == example.get("correct_answer") if evidence["final_answer"] else None}


def run_deception_example(measured_model: MeasuredModel, settings: GenerationSettings, scenario: str,
                          example: DeceptionExample, suffix: str, model_key: str,
                          grade_sandbagging: SandbaggingGrader) -> OutcomeRecord:
    name = scenario.removesuffix("_mirrored")
    generation = measured_generate(measured_model, settings, scenario, example["example_id"],
                                   deception_messages(example, name, suffix), 0)
    status = generation["status"]
    response = generation["response"] if status == "complete" else ""
    label, extra_fields = "ungraded", {}
    if name == "sandbagging":
        label, extra_fields = sandbagging_fields(example, response, grade_sandbagging)
    elif name not in ("roleplaying", "insider_trading") and status == "complete":
        label = classify(response, example.get("honest_answer", ""), example.get("deceptive_answer", ""))
    # Judge scripts prepend answer_prefix themselves.
    answer_prefix = example.get("answer_prefix", "")
    if answer_prefix and response.startswith(answer_prefix):
        response = response[len(answer_prefix):].lstrip()
    return cast(OutcomeRecord, {"example_id": example["example_id"], "scenario": scenario, "response": response,
                                "status": status, "label": label, **extra_fields,
                                "generation_turns": 1, "model_key": model_key})


def oom_outcome(scenario: str, example_id: str, model_key: str, error: harness.GenerationOOM) -> OutcomeRecord:
    return {"example_id": example_id, "scenario": scenario, "model_key": model_key, "response": "",
            "status": "error_oom", "label": "ungraded", "error": str(error)}


def write_judge_inputs(output_directory: Path, outcomes_path: Path, scenarios: list[str]) -> None:
    # Rebuild judge inputs from committed outcomes: interruption/retry cannot
    # duplicate a task or leave judges reading an unfinished generation file.
    outcomes_by_key: dict[tuple[str, str], OutcomeRecord] = {
        (outcome["scenario"], outcome["example_id"]): outcome for outcome in read_jsonl(outcomes_path)}
    for scenario in scenarios:
        suffix = "" if scenario in CODE_SCENARIOS else "_none"
        (output_directory / f"base_{scenario}{suffix}.jsonl").write_text("".join(
            json.dumps(outcome) + "\n" for (outcome_scenario, _), outcome in outcomes_by_key.items()
            if outcome_scenario == scenario))


def run_scenarios(config: RunConfig, measured_model: MeasuredModel) -> None:
    outcomes_path = config.out / "outcomes.jsonl"
    completed_keys = ({(outcome["scenario"], outcome["example_id"]) for outcome in read_jsonl(outcomes_path)}
                      if outcomes_path.exists() else set[tuple[str, str]]())
    grade_sandbagging: SandbaggingGrader = load_soo_script("reparse_sandbagging").grade_record
    sampling = sampling_kwargs(config.temperature, config.top_p, config.top_k)
    for scenario in config.scenarios:
        is_code = scenario in CODE_SCENARIOS
        settings = GenerationSettings(config.out, sampling, config.seed,
                                      config.code_max_new_tokens if is_code else config.max_new_tokens)
        examples = select_examples(read_jsonl(scenario_data_path(scenario)), config.code_n if is_code else config.n,
                                   scenario.removesuffix("_mirrored"), config.code_offset if is_code else config.offset)
        for example in examples:
            if (scenario, example["example_id"]) in completed_keys:
                continue
            try:
                outcome = (run_code_example(measured_model, settings, scenario, example, config.max_attempts, config.model)
                           if is_code else
                           run_deception_example(measured_model, settings, scenario, example, config.suffix, config.model,
                                                 grade_sandbagging))
            except harness.GenerationOOM as error:
                outcome = oom_outcome(scenario, example["example_id"], config.model, error)
            append_jsonl(outcomes_path, outcome)
            logger.info(f"OOM {scenario}/{example['example_id']}" if outcome["status"] == "error_oom"
                        else f"{scenario}/{example['example_id']}: {outcome['status']} {outcome['label']}")
    write_judge_inputs(config.out, outcomes_path, config.scenarios)
