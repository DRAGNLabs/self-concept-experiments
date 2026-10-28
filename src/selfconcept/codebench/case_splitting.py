"""Split multi-test-case inputs ("first line t, then t cases") into single-case tests.

A problem's case parser is `case_end(lines, start) -> int`: given the input's lines and the index of a
case's first line, it returns the index just past that case, asserting the shape of every line it reads
with the helpers below. Parsers are stored as source in one YAML file keyed by problem id and run with
those helpers as their only globals.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import yaml

from selfconcept.codebench.harness import WorkspaceTestCase

type CaseEnd = Callable[[list[str], int], int]


def tokens(lines: list[str], index: int, expected_count: int) -> list[str]:
    line_tokens = lines[index].split()
    assert len(line_tokens) == expected_count, f"line {index}: {len(line_tokens)} tokens, expected {expected_count}"
    return line_tokens


def ints(lines: list[str], index: int, expected_count: int) -> list[int]:
    return [int(token) for token in tokens(lines, index, expected_count)]


def string_line(lines: list[str], index: int, expected_length: int | None = None) -> str:
    text = lines[index].strip()
    assert text and " " not in text, f"line {index}: expected one string token, got {lines[index]!r}"
    assert expected_length is None or len(text) == expected_length, f"line {index}: length {len(text)}, expected {expected_length}"
    return text


def split_test(case_end: CaseEnd, test: WorkspaceTestCase) -> list[WorkspaceTestCase]:
    """One single-case test per case, each paired with its line of the answer."""
    lines = test["input"].rstrip("\n").split("\n")
    answer_lines = test["output"].rstrip("\n").split("\n")
    (case_count,) = ints(lines, 0, 1)
    assert len(answer_lines) == case_count, f"{len(answer_lines)} answer lines for {case_count} cases"
    cases: list[WorkspaceTestCase] = []
    start = 1
    for answer_line in answer_lines:
        end = case_end(lines, start)
        cases.append({"input": "1\n" + "\n".join(lines[start:end]) + "\n", "output": answer_line + "\n"})
        start = end
    assert start == len(lines), f"{len(lines) - start} unconsumed lines after {case_count} cases"
    return cases


def load_case_parsers(path: Path) -> dict[str, CaseEnd]:
    source_by_problem_id: dict[str, str] = yaml.safe_load(path.read_text())
    return {problem_id: compile_case_parser(source) for problem_id, source in source_by_problem_id.items()}


def compile_case_parser(source: str) -> CaseEnd:
    parser_globals: dict[str, object] = {"ints": ints, "tokens": tokens, "string_line": string_line}
    exec(source, parser_globals)
    return cast(CaseEnd, parser_globals["case_end"])
