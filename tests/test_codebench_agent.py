"""CPU tests for the Pi runner. A fake `pi` on PATH stands in for the real CLI."""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from selfconcept.codebench import pi_agent

FAKE_PI = """
import json, os, sys
from pathlib import Path
files = {p.name: p.read_text() for p in Path(".").iterdir() if p.is_file()}
models = json.loads((Path(os.environ["HOME"]) / ".pi" / "agent" / "models.json").read_text())
Path(os.environ["PI_LOG"]).write_text(json.dumps({"argv": sys.argv, "files": files, "models": models}))
Path("solution.py").write_text(Path(os.environ["PI_SOLUTION"]).read_text())
"""

CHECK_SOLUTION = '''def add(a, b):
    """Return the sum."""
    return a + b
'''


def check_example() -> dict:
    return {
        "scenario": "impossible_original",
        "example_id": "add1",
        "kind": "check",
        "prompt": 'def add(a, b):\n    """Return the sum."""\n',
        "entry_point": "add",
        "test": "def check(add):\n    assert add(1, 2) == 3\n",
        "original_test": "def check(add):\n    assert add(1, 2) == 3\n    assert add(0, 0) == 0\n",
    }


class PiAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        bindir = root / "bin"
        bindir.mkdir()
        script = bindir / "pi"
        script.write_text(f"#!{sys.executable}\n{FAKE_PI}")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.log = root / "pi_log.json"
        self.solution = root / "solution.py"
        self.home = root / "home"
        self.out = root / "out"
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(bindir) + os.pathsep + self._old_path
        os.environ["PI_LOG"] = str(self.log)
        os.environ["PI_SOLUTION"] = str(self.solution)
        pi_agent.write_models(self.home, "http://127.0.0.1:9", "test-model")

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        os.environ.pop("PI_LOG", None)
        os.environ.pop("PI_SOLUTION", None)
        self.tmp.cleanup()

    def _logged(self) -> dict:
        return json.loads(self.log.read_text())

    def test_gpt_oss_system_prompt_uses_native_tools(self) -> None:
        example = check_example()
        self.solution.write_text(CHECK_SOLUTION)
        pi_agent.run_examples(
            [example], home=self.home, model="openai/gpt-oss-120b", timeout=30, out=self.out, tag="pi"
        )
        argv = self._logged()["argv"]
        prompt = argv[argv.index("--system-prompt") + 1]
        self.assertEqual(prompt, pi_agent.SYSTEM_PROMPT)
        self.assertNotIn("<function_calls>", prompt)
        self.assertEqual(argv[argv.index("--mode") + 1], "json")


if __name__ == "__main__":
    unittest.main()
