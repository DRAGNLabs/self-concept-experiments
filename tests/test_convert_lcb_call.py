"""The pure helpers of convert.convert_lcb_call: stub extraction, docstring, case parsing, check rendering."""
import unittest

from selfconcept.codebench.convert import lcb_call_case, lcb_check, lcb_docstring, lcb_function_stub
from selfconcept.codebench.harness import check_program


class LcbCallHelpersTest(unittest.TestCase):
    def test_stub_strips_self_and_imports_typing_names(self):
        starter = "class Solution:\n    def count(self, nums: List[int], k: int) -> int:\n        "
        self.assertEqual(
            lcb_function_stub(starter),
            ("count", "def count(nums: List[int], k: int) -> int:", "from typing import List"),
        )

    def test_stub_rejects_trees_and_multiple_methods(self):
        self.assertIsNone(lcb_function_stub("class Solution:\n    def f(self, root: Optional[TreeNode]) -> int:\n"))
        self.assertIsNone(lcb_function_stub("class A:\n    def f(self):\n        pass\n    def g(self):\n        pass\n"))

    def test_case_parses_one_json_value_per_line(self):
        self.assertEqual(lcb_call_case({"input": '"ab"\n[1, 2]\n3', "output": "true"}), (["ab", [1, 2], 3], True))
        self.assertIsNone(lcb_call_case({"input": "not json", "output": "1"}))

    def test_prompt_and_check_compile_and_run(self):
        prompt = "from typing import List\n\ndef total(nums: List[int]) -> int:\n" + lcb_docstring(
            'Return the sum.\n\nExample 1:\n\nInput: nums = [1,2]\nOutput: 3\n\nConstraints:\n\n"""quoted""" and a slash \\'
        )
        test = lcb_check([([[1, 2]], 3)], [([[]], 0), ([[5]], 5)])
        example = {"prompt": prompt, "entry_point": "total", "test": test, "original_test": test}
        program = check_program(example, "def total(nums):\n    return sum(nums)\n", test, tests_first=False)
        namespace: dict = {}
        exec(compile(program, "<verify>", "exec"), namespace)  # raises if the stub, docstring or asserts are broken
        self.assertIn("# Example test cases\n    assert candidate([1, 2]) == 3\n    # Additional test cases", test)
        self.assertIn('    """ Return the sum.\n\n    Example 1:', prompt)


if __name__ == "__main__":
    unittest.main()
