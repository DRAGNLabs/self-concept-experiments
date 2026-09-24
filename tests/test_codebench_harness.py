"""CPU tests for the EvilGenie workspace prompt (selfconcept.codebench.harness).

Regression for the 2026-09-24 defect: the shown workspace files went through
the 3000-char tool-output truncation, so test.py (4.2-5.1 KB) was always cut in
the middle and test_cases.json was cut on most problems.
"""

import json
import unittest

from selfconcept.codebench import harness
from selfconcept.codebench.harness import (
    WORKSPACE_TESTS_MAX_CHARS,
    load_examples,
    run_workspace_example,
    show_test_cases,
    workspace_files,
    workspace_prompt,
)


def _example(is_function: bool, n_tests: int, chars_per_test: int = 10) -> dict:
    tests = [{"input": "x" * chars_per_test, "output": str(i)} for i in range(n_tests)]
    return {
        "scenario": "evilgenie",
        "example_id": f"t_{'fn' if is_function else 'io'}",
        "kind": "workspace",
        "title": "T",
        "description": "Print the answer.",
        "is_function": is_function,
        "function_name": "solve" if is_function else None,
        "starter_code": "def solve(a):\n    pass\n" if is_function else "",
        "visible_tests": tests,
        "holdout_tests": tests[:1],
    }


class WorkspacePromptTest(unittest.TestCase):
    def test_runner_and_solution_shown_verbatim(self):
        for is_function in (True, False):
            ex = _example(is_function, 3)
            originals = workspace_files(ex)
            prompt, shown = workspace_prompt(ex, originals)
            self.assertGreater(len(originals["test.py"]), harness.TOOL_RESULT_MAX_CHARS)
            self.assertIn("### test.py\n```\n" + originals["test.py"] + "\n```", prompt)
            self.assertIn("### solution.py\n```\n" + originals["solution.py"] + "\n```", prompt)
            self.assertIn("### test_cases.json\n```\n" + originals["test_cases.json"] + "\n```", prompt)
            self.assertNotIn("characters truncated", prompt)
            self.assertEqual(shown, 3)

    def test_show_test_cases_keeps_whole_entries(self):
        ex = _example(False, 40, chars_per_test=1000)
        text = workspace_files(ex)["test_cases.json"]
        self.assertGreater(len(text), WORKSPACE_TESTS_MAX_CHARS)
        view, shown = show_test_cases(text)
        body, _, note = view.partition("\n\n... [")
        kept = json.loads(body)  # whole entries: the shown part is valid JSON
        self.assertEqual(len(kept), shown)
        self.assertEqual(kept, json.loads(text)[:shown])
        self.assertGreaterEqual(shown, 5)
        self.assertLess(shown, 40)
        self.assertLessEqual(len(body), WORKSPACE_TESTS_MAX_CHARS)
        self.assertIn(f"{40 - shown} more test cases omitted", note)
        self.assertIn("holds all 40", note)

    def test_show_test_cases_oversized_first_entry(self):
        ex = _example(False, 2, chars_per_test=30_000)
        view, shown = show_test_cases(workspace_files(ex)["test_cases.json"])
        self.assertEqual(shown, 1)
        body = view.partition("\n\n... [")[0]
        self.assertLess(len(body), WORKSPACE_TESTS_MAX_CHARS)
        self.assertIn("characters truncated", body)  # the entry's strings, not the file
        self.assertIn("1 more test cases omitted", view)

    def test_dataset_prompts(self):
        for ex in load_examples("evilgenie"):
            originals = workspace_files(ex)
            prompt, shown = workspace_prompt(ex, originals)
            self.assertIn("### test.py\n```\n" + originals["test.py"] + "\n```", prompt, ex["example_id"])
            self.assertGreaterEqual(shown, 1)
            self.assertLessEqual(shown, len(ex["visible_tests"]))
            self.assertLess(len(prompt), WORKSPACE_TESTS_MAX_CHARS + 12_000, ex["example_id"])

    def test_end_to_end_records_shown_counts(self):
        ex = _example(False, 2)
        seen = []

        def generate(messages, turn, example_id=None):
            seen.append(messages[0]["content"])
            return "```python filename=solution.py\nimport sys\nprint(sys.stdin.read().count('x') and 0)\n```", False

        record = run_workspace_example(ex, generate, max_attempts=1)
        self.assertEqual(record["tests_shown"], 2)
        self.assertEqual(record["n_visible_tests"], 2)
        self.assertIn(workspace_files(ex)["test.py"], seen[0])
        self.assertIn(record["label"], {"solved", "failed", "hack_visible_only"})


if __name__ == "__main__":
    unittest.main()
