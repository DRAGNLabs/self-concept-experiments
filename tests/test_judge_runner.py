"""Pure-function tests for the headless judge runner (no CLI call, no network)."""
import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "judge_runner",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "deceiver-axis", "scripts", "judge_runner.py"),
)
jr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(jr)


def _row(name, verdict="NOT_HACK", strength=None, mechanism="none", pressure="none_stated", fc="no"):
    return {"file": name, "verdict": verdict, "strength": strength, "mechanism": mechanism, "pressure_source": pressure, "feedback_confusion": fc}


class JudgeRunnerTest(unittest.TestCase):
    def test_parse_task(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "JUDGE_TASK_proxy.md")
            open(p, "w").write("x. Your assignment: batch list /a/b/dossiers/batches/batch_N.list. Dossier paths ... "
                               "Write exactly one output file: /a/b/verdicts/proxy_batches/set_batch_N.jsonl, one JSON object")
            self.assertEqual(jr.parse_task(p), ("/a/b/dossiers/batches/", "/a/b/verdicts/proxy_batches/", "set_"))
            open(p, "w").write("no paths here")
            with self.assertRaises(ValueError):
                jr.parse_task(p)

    def test_parse_spec(self):
        self.assertEqual(jr.parse_spec("all", [3, 1, 2]), [1, 2, 3])
        self.assertEqual(jr.parse_spec("0-2,5", range(10)), [0, 1, 2, 5])
        with self.assertRaises(ValueError):
            jr.parse_spec("0-3", [0, 1])

    def test_check_output(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "batch_0.jsonl")
            self.assertEqual(jr.check_output(p, ["a.md"])[0], "missing")
            open(p, "w").write("{not json\n")
            self.assertEqual(jr.check_output(p, ["a.md"])[0], "invalid")
            open(p, "w").write(json.dumps(_row("b.md")) + "\n")
            self.assertEqual(jr.check_output(p, ["a.md"])[0], "invalid")
            open(p, "w").write(json.dumps(_row("a.md")) + "\n" + json.dumps(_row("b.md", "HACK", "STRONG", "lookup_table")) + "\n")
            self.assertEqual(jr.check_output(p, ["a.md", "b.md"]), ("ok", ""))
            self.assertEqual(jr.check_output(p, ["b.md", "a.md"])[0], "invalid")
            open(p, "w").write(json.dumps(_row("a.md", "HACK", None, "lookup_table")) + "\n")
            self.assertEqual(jr.check_output(p, ["a.md"])[0], "bad_fields")
            open(p, "w").write(json.dumps(_row("a.md", pressure="deadline")) + "\n")
            self.assertEqual(jr.check_output(p, ["a.md"])[0], "bad_fields")

    def test_read_audit(self):
        names = ["aaa111.md", "bbb222.md"]
        cmds = [{"command": "wc -c aaa111.md bbb222.md", "result_chars": 40, "is_error": False},
                {"command": "cd /x && cat aaa111.md bbb222.md", "result_chars": 1000, "is_error": False}]
        a = jr.read_audit(cmds, 990, names)
        self.assertEqual((a["read_chars"], a["short"], a["filter_cmds"]), (1000, False, []))
        self.assertTrue(jr.read_audit(cmds, 1500, names)["short"])
        cmds.append({"command": "python3 -I - <<'EOF'\nopen('out.jsonl','w').write('aaa111.md')\nEOF", "result_chars": 70, "is_error": False})
        self.assertEqual(jr.read_audit(cmds, 990, names)["read_chars"], 1000)
        cmds.append({"command": "grep -n hardcode aaa111.md", "result_chars": 50, "is_error": False})
        self.assertEqual(len(jr.read_audit(cmds, 990, names)["filter_cmds"]), 1)
        via_list = [{"command": "cd /d; cat batches/batch_3.list | xargs wc -c", "result_chars": 400, "is_error": False},
                    {"command": "cd /d; cat $(cat batches/batch_3.list)", "result_chars": 2000, "is_error": False}]
        self.assertTrue(jr.read_audit(via_list, 1990, names)["short"])
        self.assertFalse(jr.read_audit(via_list, 1990, names, 3)["short"])
        big = [{"command": "head -c 120000 aaa111.md", "result_chars": 120000, "is_error": False},
               {"command": "tail -c +120001 aaa111.md", "result_chars": 5000, "is_error": False}]
        self.assertEqual(jr.read_audit(big, 125000, ["aaa111.md"])["filter_cmds"], [])


if __name__ == "__main__":
    unittest.main()
