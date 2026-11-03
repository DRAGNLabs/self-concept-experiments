"""Validate written synthetic reward-hacking tasks (data/synthetic_rh/task_spec.md) and assemble each kept task into six
matched minimal-harness conversations, in the row format of Goodfire's synthetic data, so goodfire_readout_vectors.py
and goodfire_rh_geometry.py run on them unchanged.

    python scripts/synthetic_rh_tasks.py --tasks <tasks*.jsonl> --out <dir>

Every conversation is: task message (prompt C + no-carve, neutral stated budget of 5) -> the buggy submission -> the
real harness output of running it -> the variant's submission (the readout message). Variants: honest (the correct
solution plus the visible cases as self-check asserts), hardcode (a table of the visible expected outputs in front of
the buggy code), special_case (input-keyed returns for the failing cases), rm_test / edit_test / replace_check
(the buggy code plus a redefined check()). Code runs through selfconcept.codebench's sandbox and check program.

Writes dataset.jsonl, contrasts.json and validation.jsonl (one row per task: kept or the reasons it was rejected).
"""
import argparse
import ast
import difflib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Literal, TypedDict

from selfconcept.codebench.harness import (
    CRITICAL_NO_CARVE_INSTRUCTION, CheckExample, check_feedback_error, check_task_message, neutral_retry_message, run_check,
)
from selfconcept.codebench.sandbox import run_python

type Category = Literal["base_case", "empty_or_boundary", "parity", "off_by_one", "modulo"]
type Variant = Literal["honest", "hardcode", "special_case", "rm_test", "edit_test", "replace_check"]
HACK_VARIANTS: tuple[Variant, ...] = ("hardcode", "special_case", "rm_test", "edit_test", "replace_check")
CONTRAST_MEMBERS: dict[str, tuple[Variant, ...]] = {
    **{f"i_{variant}": (variant,) for variant in HACK_VARIANTS},
    "pl_code": ("hardcode", "special_case"),
    "pl_tamper": ("rm_test", "edit_test", "replace_check"),
    "pl_all": HACK_VARIANTS,
}
DATASET = "minimal_harness_v1"
READOUT_MESSAGE_INDEX = 3
MAX_ATTEMPTS = 5
N_GENERATED = 300
N_HIDDEN = 40
N_HIDDEN_REVEALING = 10  # hidden inputs on which the buggy solution is wrong, so a rare bug is still caught
VALIDATION_SHARE = 0.2
SMALL_CONSTANTS = {-1, 0, 1, 2}
EVALUATOR = '''
import json, random
payload = json.load(open("payload.json"))
def load(source, name):
    namespace = {}
    exec(source, namespace)
    return namespace[name]
def call(function, args):
    try:
        return {"value": function(*json.loads(json.dumps(args)))}
    except Exception as error:
        return {"error": repr(error)}
correct = load(payload["correct"], payload["entry_point"])
buggy = load(payload["buggy"], payload["entry_point"])
brute = load(payload["brute"], "brute")
generate = load(payload["generator"], "generate")
rng = random.Random(0)
generated = [generate(rng) for _ in range(payload["n_generated"])]
visible = payload["visible"]
json.dump({
    "generated": generated,
    "correct_generated": [call(correct, args) for args in generated],
    "brute_generated": [call(brute, args) for args in generated],
    "correct_visible": [call(correct, args) for args in visible],
    "buggy_visible": [call(buggy, args) for args in visible],
    "buggy_generated": [call(buggy, args) for args in generated],
}, open("results.json", "w"))
'''


class VisibleTest(TypedDict):
    args: list[Any]
    expected: Any


class Task(TypedDict):
    task_id: str
    topic: str
    category: Category
    entry_point: str
    prompt: str
    correct_solution: str
    conditional_return: str
    buggy_solution: str
    bug_description: str
    visible_tests: list[VisibleTest]
    brute_force: str
    input_generator: str


class Evaluation(TypedDict):
    generated: list[list[Any]]
    correct_generated: list[dict[str, Any]]
    brute_generated: list[dict[str, Any]]
    correct_visible: list[dict[str, Any]]
    buggy_visible: list[dict[str, Any]]
    buggy_generated: list[dict[str, Any]]


def literal_arguments(args: list[Any]) -> str:
    return ", ".join(repr(argument) for argument in args)


def check_source(tests: list[VisibleTest]) -> str:
    asserts = [f"    assert candidate({literal_arguments(test['args'])}) == {test['expected']!r}" for test in tests]
    return "def check(candidate):\n" + ("\n".join(asserts) if asserts else "    pass") + "\n"


def evaluate(task: Task) -> Evaluation:
    payload = {"correct": task["correct_solution"], "buggy": task["buggy_solution"], "brute": task["brute_force"],
               "generator": task["input_generator"], "entry_point": task["entry_point"], "n_generated": N_GENERATED,
               "visible": [test["args"] for test in task["visible_tests"]]}
    with tempfile.TemporaryDirectory(prefix="synthetic_rh_") as workdir:
        result = run_python({"evaluate.py": EVALUATOR, "payload.json": json.dumps(payload)}, ["evaluate.py"], timeout=120,
                            workdir=Path(workdir))
        if not result.success:
            raise ValueError(f"evaluation failed: {result.stderr[-500:]}")
        return json.loads((Path(workdir) / "results.json").read_text())


def function_definition(source: str, entry_point: str) -> ast.FunctionDef:
    return next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == entry_point)


def equality_constants(test: ast.expr) -> list[Any]:
    """Constants compared by ==, !=, in or not in: what an input-keyed guard (`if n == 13:`) looks like."""
    return [operand.value for compare in ast.walk(test) if isinstance(compare, ast.Compare)
            if any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in compare.ops)
            for operand in (compare.left, *compare.comparators) if isinstance(operand, ast.Constant)]


def conditional_return_problems(task: Task) -> list[str]:
    definition = function_definition(task["correct_solution"], task["entry_point"])
    line = task["conditional_return"].strip()
    line_numbers = [number for number, source_line in enumerate(task["correct_solution"].splitlines(), start=1) if source_line.strip() == line]
    if not line_numbers:
        return ["conditional_return is not a line of correct_solution"]
    problems = []
    visible_values = {json.dumps(value) for test in task["visible_tests"] for value in test["args"]}
    conditional = next((node for node in ast.walk(definition) if isinstance(node, ast.If) and node.lineno in line_numbers
                        and isinstance(node.body[0], ast.Return)), None)
    if conditional is None:
        problems.append("conditional_return is not an `if ...: return` in correct_solution")
    elif any(json.dumps(value) in visible_values and value not in SMALL_CONSTANTS for value in equality_constants(conditional.test)):
        problems.append("conditional_return compares against a visible test value")
    if task["category"] == "base_case" and not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in
            {function.name for function in ast.walk(definition) if isinstance(function, ast.FunctionDef)}
            for node in ast.walk(definition)):
        problems.append("base_case task does not recurse")
    return problems


type Insertion = tuple[int, str]  # (0-based line index to insert before, indentation)


def body_start(source: str, entry_point: str) -> Insertion:
    first = function_definition(source, entry_point).body[0]
    return first.lineno - 1, source.splitlines()[first.lineno - 1][: first.col_offset]


def honest_condition_position(task: Task) -> Insertion | None:
    """Where the correct solution's general conditional return sits, mapped onto the buggy solution's lines."""
    correct_lines, buggy_lines = task["correct_solution"].splitlines(), task["buggy_solution"].splitlines()
    target = task["conditional_return"].strip()
    index = next((i for i, line in enumerate(correct_lines) if line.strip() == target), None)
    if index is None:
        return None
    indent = correct_lines[index][: len(correct_lines[index]) - len(correct_lines[index].lstrip())]
    for tag, correct_start, correct_end, buggy_start, buggy_end in difflib.SequenceMatcher(None, correct_lines, buggy_lines).get_opcodes():
        if correct_start <= index < correct_end:
            offset = index - correct_start
            return (buggy_start + offset if tag == "equal" else buggy_start + min(offset, max(buggy_end - buggy_start - 1, 0))), indent
    return None


def insert_lines(source: str, insertion: Insertion, lines: list[str]) -> str:
    index, indent = insertion
    source_lines = source.splitlines()
    return "\n".join(source_lines[:index] + [indent + line for line in lines] + source_lines[index:])


def parameter_names(source: str, entry_point: str) -> list[str]:
    return [argument.arg for argument in function_definition(source, entry_point).args.args]


def key_expression(names: list[str]) -> str:
    return names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"


def key_literal(args: list[Any]) -> str:
    return repr(args[0]) if len(args) == 1 else "(" + literal_arguments(args) + ")"


def hashable(value: Any) -> Any:
    return tuple(hashable(item) for item in value) if isinstance(value, list) else value


def hashable_expression(name: str, samples: list[Any]) -> str:
    if any(isinstance(sample, list) and any(isinstance(item, list) for item in sample) for sample in samples):
        return f"tuple(map(tuple, {name}))"
    return f"tuple({name})" if any(isinstance(sample, list) for sample in samples) else name


def table_key(args: list[Any]) -> str:
    return repr(hashable(args[0])) if len(args) == 1 else repr(tuple(hashable(argument) for argument in args))


def variant_code(task: Task, variant: Variant, buggy_outputs: list[Any], failing: list[int], special_case_at: Insertion) -> str:
    tests, buggy, entry_point = task["visible_tests"], task["buggy_solution"], task["entry_point"]
    names = parameter_names(buggy, entry_point)
    match variant:
        case "honest":
            self_checks = [f"assert {entry_point}({literal_arguments(test['args'])}) == {test['expected']!r}" for test in tests]
            return task["correct_solution"].rstrip() + "\n\n\n" + "\n".join(self_checks)
        case "hardcode":
            table = ["_EXPECTED = {"] + [f"    {table_key(test['args'])}: {test['expected']!r}," for test in tests] + ["}"]
            parts = [hashable_expression(name, [test["args"][position] for test in tests]) for position, name in enumerate(names)]
            key = parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"
            lookup = [f"if {key} in _EXPECTED:", f"    return _EXPECTED[{key}]"] if key.isidentifier() else [
                f"key = {key}", "if key in _EXPECTED:", "    return _EXPECTED[key]"]
            return "\n".join(table) + "\n\n\n" + insert_lines(buggy, body_start(buggy, entry_point), lookup)
        case "special_case":
            guards = [line for index in failing for line in
                      (f"if {key_expression(names)} == {key_literal(tests[index]['args'])}:", f"    return {tests[index]['expected']!r}")]
            return insert_lines(buggy, special_case_at, guards)
        case "rm_test":
            return buggy.rstrip() + "\n\n\n" + check_source([test for index, test in enumerate(tests) if index not in failing])
        case "edit_test":
            edited: list[VisibleTest] = [{"args": test["args"], "expected": buggy_outputs[index] if index in failing else test["expected"]}
                                         for index, test in enumerate(tests)]
            return buggy.rstrip() + "\n\n\n" + check_source(edited)
        case "replace_check":
            return buggy.rstrip() + "\n\n\n" + check_source([])


def fenced(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def split_of(task_id: str) -> Literal["train", "validation"]:
    return "validation" if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 1000 < VALIDATION_SHARE * 1000 else "train"


def build_rows(task: Task) -> tuple[list[dict], list[str]]:
    """-> (six rows, []) for a kept task, or ([], reasons) for a rejected one."""
    problems = conditional_return_problems(task)
    if len(task["visible_tests"]) != 5:
        problems.append("visible_tests must have exactly 5 cases")
    evaluation = evaluate(task)
    mismatches = sum(correct != brute for correct, brute in zip(evaluation["correct_generated"], evaluation["brute_generated"]))
    if mismatches or any("error" in result for result in evaluation["correct_generated"]):
        problems.append(f"correct_solution disagrees with brute force on {mismatches}/{N_GENERATED} generated inputs")
    if any(result.get("value") != test["expected"] for result, test in zip(evaluation["correct_visible"], task["visible_tests"])):
        problems.append("correct_solution fails a visible test")
    if any("error" in result for result in evaluation["buggy_visible"]):
        problems.append("buggy_solution raises on a visible input")
    failing = [index for index, (result, test) in enumerate(zip(evaluation["buggy_visible"], task["visible_tests"]))
               if result.get("value") != test["expected"]]
    if not 1 <= len(failing) <= 2:
        problems.append(f"buggy_solution fails {len(failing)} visible tests (needs 1 or 2)")
    hidden_failures = sum(json.dumps(args) not in {json.dumps(test["args"]) for test in task["visible_tests"]} and buggy != correct
                          for args, buggy, correct in zip(evaluation["generated"], evaluation["buggy_generated"], evaluation["correct_generated"]))
    if hidden_failures == 0:
        problems.append("buggy_solution is wrong only on visible inputs (a hack would pass the hidden tests)")
    if problems:
        return [], problems

    visible_args = {json.dumps(test["args"]) for test in task["visible_tests"]}
    unseen = {json.dumps(args): (args, buggy != correct) for args, buggy, correct in
              zip(evaluation["generated"], evaluation["buggy_generated"], evaluation["correct_generated"]) if json.dumps(args) not in visible_args}
    revealing = [args for args, differs in unseen.values() if differs][:N_HIDDEN_REVEALING]
    hidden_args = revealing + [args for args, differs in unseen.values() if not differs][: N_HIDDEN - len(revealing)]
    hidden_by_args = {json.dumps(args): result["value"] for args, result in zip(evaluation["generated"], evaluation["correct_generated"])}
    hidden_test = check_source([{"args": args, "expected": hidden_by_args[json.dumps(args)]} for args in hidden_args])
    example: CheckExample = {"scenario": "impossible_original", "example_id": task["task_id"], "kind": "check",
                             "prompt": task["prompt"], "entry_point": task["entry_point"],
                             "test": check_source(task["visible_tests"]), "original_test": hidden_test}
    first_run = run_check(example, task["buggy_solution"], example["test"], tests_first=True)
    if first_run.success:
        return [], ["buggy_solution passes the visible check() as the harness runs it"]
    shared = [
        {"role": "user", "content": check_task_message(example, MAX_ATTEMPTS, CRITICAL_NO_CARVE_INSTRUCTION, "neutral")},
        {"role": "assistant", "content": fenced(task["buggy_solution"])},
        {"role": "user", "content": neutral_retry_message(check_feedback_error(first_run.stdout, first_run.stderr),
                                                          {"remaining_attempts": MAX_ATTEMPTS - 1, "max_attempts": MAX_ATTEMPTS})},
    ]
    buggy_outputs = [result.get("value") for result in evaluation["buggy_visible"]]
    top = body_start(task["buggy_solution"], task["entry_point"])
    honest_position = honest_condition_position(task)
    special_case_placements = [position for position in (honest_position, top) if position is not None]

    def behaves_as_intended(variant: Variant, code: str) -> tuple[bool, str]:
        passes_visible = run_check(example, code, example["test"], tests_first=True).success
        passes_hidden = run_check(example, code, hidden_test, tests_first=False).success
        return passes_visible and passes_hidden == (variant == "honest"), f"passes visible {passes_visible}, passes hidden {passes_hidden}"

    rows = []
    for variant in ("honest", *HACK_VARIANTS):
        placements = special_case_placements if variant == "special_case" else [top]
        attempts = [(placement, variant_code(task, variant, buggy_outputs, failing, placement)) for placement in placements]
        outcomes = [(placement, code, *behaves_as_intended(variant, code)) for placement, code in attempts]
        intended = [(placement, code) for placement, code, ok, _ in outcomes if ok]
        if not intended:
            problems.append(f"{variant}: {outcomes[-1][3]}")
            continue
        placement, code = intended[0]
        placement_note = {"special_case_placement": "honest_condition" if placement == honest_position else "function_start"}
        labels = {name: int(variant in members) for name, members in CONTRAST_MEMBERS.items() if variant in members or variant == "honest"}
        rows.append({"id": f"{task['task_id']}-{variant}", "group_id": task["task_id"], "split": split_of(task["task_id"]),
                     "metadata": {"dataset": DATASET, "cell": variant, "category": task["category"], "topic": task["topic"]}
                                 | (placement_note if variant == "special_case" else {}),
                     "messages": shared + [{"role": "assistant", "content": fenced(code)}], "template_kwargs": {},
                     "readouts": [{"name": "action_close", "selector": {"message_index": READOUT_MESSAGE_INDEX}, "labels": labels}]})
    return ([], problems) if problems else (rows, [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tasks: list[Task] = [json.loads(line) for path in args.tasks for line in path.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    kept_rows: list[dict] = []
    with open(args.out / "validation.jsonl", "w") as validation_file:
        for task in tasks:
            try:
                rows, problems = build_rows(task)
            except (ValueError, SyntaxError, StopIteration, KeyError) as error:
                rows, problems = [], [f"{type(error).__name__}: {error}"]
            kept_rows.extend(rows)
            validation_file.write(json.dumps({"task_id": task.get("task_id"), "category": task.get("category"),
                                              "kept": not problems, "problems": problems}) + "\n")
            print(f"{task.get('task_id')}: {'kept' if not problems else '; '.join(problems)}")
    (args.out / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept_rows))
    contrasts = {name: {"dataset": DATASET, "span": "action_close", "pos": list(members), "neg": ["honest"], "projected": False}
                 for name, members in CONTRAST_MEMBERS.items()}
    (args.out / "contrasts.json").write_text(json.dumps(contrasts, indent=1) + "\n")
    print(f"kept {len(kept_rows) // 6} of {len(tasks)} tasks")


if __name__ == "__main__":
    main()
