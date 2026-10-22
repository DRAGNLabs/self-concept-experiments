"""CPU tests for the EvilGenie workspace prompt (selfconcept.codebench.harness).

Regression for the 2026-09-24 defect: the shown workspace files went through
the 3000-char tool-output truncation, so test.py (4.2-5.1 KB) was always cut in
the middle and test_cases.json was cut on most problems.
"""

import unittest

from selfconcept.codebench import harness
from selfconcept.codebench.harness import (
    load_examples,
    parse_file_blocks,
    run_workspace_example,
    workspace_files,
    workspace_task_message,
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
    def test_files_shown_verbatim(self):
        for is_function in (True, False):
            ex = _example(is_function, 3)
            originals = workspace_files(ex)
            prompt = workspace_task_message(ex, originals)
            self.assertGreater(len(originals["test.py"]), harness.TOOL_RESULT_MAX_CHARS)
            for name in ("solution.py", "test.py", "test_cases.json"):
                self.assertIn(originals[name], prompt)
            self.assertNotIn("characters truncated", prompt)
            # The shown files are themselves tagged file blocks the model can copy.
            self.assertEqual(parse_file_blocks(prompt)["test.py"].rstrip("\n"), originals["test.py"].rstrip("\n"))

    def test_dataset_prompts_show_runner_and_tests(self):
        for ex in load_examples("evilgenie"):
            originals = workspace_files(ex)
            prompt = workspace_task_message(ex, originals)
            self.assertIn(originals["test.py"], prompt, ex["example_id"])
            self.assertIn(originals["test_cases.json"], prompt, ex["example_id"])

    def test_end_to_end_runs_test_py(self):
        ex = _example(False, 2)
        seen = []

        def generate(messages, turn, example_id=None):
            seen.append(messages[0]["content"])
            return "```python filename=solution.py\nimport sys\nprint(sys.stdin.read().count('x') and 0)\n```", False

        record = run_workspace_example(ex, generate, max_attempts=1)
        self.assertIn(workspace_files(ex)["test.py"], seen[0])
        self.assertIn(record["label"], {"solved", "failed", "hack_visible_only"})
        self.assertFalse(record["test_modified"])


if __name__ == "__main__":
    unittest.main()
