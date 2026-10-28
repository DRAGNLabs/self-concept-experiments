"""CPU tests for the IFBench task of experiments/schemer-axis/scripts/capability_eval_vllm.py: the loose-accuracy
variants and the verdict string, against a fake `ifbench.instructions_registry`; one test runs the real verifiers when
the `ifbench` package is installed."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/schemer-axis/scripts/capability_eval_vllm.py"


def load_script():
    spec = importlib.util.spec_from_file_location("capability_eval_vllm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakeChecker:
    """Follows when the response contains the keyword; `needs_prompt` checkers ask for the prompt like IFEval's do."""

    needs_prompt = False

    def __init__(self, instruction_id):
        self.instruction_id = instruction_id
        self.keyword = None
        self.prompt = None

    def build_description(self, keyword=None, prompt=None):
        if keyword is not None:
            self.keyword = keyword
        if prompt is not None:
            self.prompt = prompt

    def get_instruction_args(self):
        return {"prompt": None} if self.needs_prompt else {"keyword": self.keyword}

    def check_following(self, value):
        assert not self.needs_prompt or self.prompt is not None
        return self.keyword in value


class PromptChecker(FakeChecker):
    needs_prompt = True


class FakeRegistryTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: sys.modules.get(k) for k in ("ifbench", "ifbench.instructions_registry")}
        registry = types.ModuleType("ifbench.instructions_registry")
        registry.INSTRUCTION_DICT = {"fake:keyword": FakeChecker, "fake:prompt": PromptChecker}
        package = types.ModuleType("ifbench")
        package.instructions_registry = registry
        sys.modules["ifbench"] = package
        sys.modules["ifbench.instructions_registry"] = registry
        self.m = load_script()
        self.item = {"id": "ifbench/1", "prompt": "Write about cats.", "instruction_ids": ["fake:keyword", "fake:prompt"],
                     "instruction_kwargs": [{"keyword": "cat"}, {"keyword": "dog"}]}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_verdict_string_and_prompt_level_accuracy(self):
        self.assertEqual(self.m.score_ifbench(self.item, "A cat and a dog."), ("1,1", True))
        self.assertEqual(self.m.score_ifbench(self.item, "A cat alone."), ("1,0", False))
        self.assertEqual(self.m.score_ifbench(self.item, "   "), (None, False))

    def test_loose_variants(self):
        # the keyword only in the first line: dropping the first line fails, but the as-is variant passes -> loose follows
        self.assertEqual(self.m.ifbench_followed(self.item, "cat\nsomething\ndog"), [True, True])
        # keyword hidden by asterisks only passes after the asterisk-stripped variant
        self.assertEqual(self.m.ifbench_followed(self.item, "c*a*t and d*o*g"), [True, True])

    def test_task_summary_has_instruction_accuracy(self):
        self.assertIn("ifbench", self.m.TASKS)


@unittest.skipUnless(importlib.util.find_spec("ifbench"), "ifbench package not installed")
class RealRegistryTest(unittest.TestCase):
    def test_real_verifier(self):
        m = load_script()
        from ifbench import instructions_registry

        key = "keywords:existence"
        self.assertIn(key, instructions_registry.INSTRUCTION_DICT)
        item = {"id": "ifbench/x", "prompt": "Tell me a story.", "instruction_ids": [key],
                "instruction_kwargs": [{"keywords": ["lantern", "river"]}]}
        self.assertEqual(m.score_ifbench(item, "The lantern floated down the river."), ("1", True))
        self.assertEqual(m.score_ifbench(item, "The lantern stayed home."), ("0", False))


if __name__ == "__main__":
    unittest.main()
