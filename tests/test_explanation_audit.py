"""Tests for the explanation-mismatch audit bundle and its analysis (synthetic episodes, no model, no CLI call)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "explanation_audit",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "deceiver-axis", "scripts", "explanation_audit.py"),
)
ea = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ea)


def _attempt(answer: str, success: bool) -> dict:
    return dict(completion=f"Here is my reply.\n```python\n{answer}\n```" if answer else "The tests are wrong.", answer=answer, success=success)


def _write_arm(root: Path, arm: str, problems: int = 8, seeds: int = 2) -> Path:
    for seed in range(seeds):
        path = root / arm / f"seed{seed}"
        path.mkdir(parents=True)
        rows = []
        for p in range(problems):
            attempts = [_attempt("def f(): return 1", False), _attempt("" if p == 0 else "def f(): return 2", p != 0)]
            if p == 1:
                attempts = [_attempt("", False)]  # no code in any turn: not eligible
            rows.append(dict(example_id=f"prob_{p}", attempts=attempts))
        (path / "main_impossible_original.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (path / "main_x_reasoning.jsonl").write_text('{"not": "an episode"}\n')
    return root / arm


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        spec = {"toy": dict(name="Toy", alpha=0.2, arms={arm: str(_write_arm(root / "runs", arm)) for arm in ea.ARMS})}
        (root / "arms.json").write_text(json.dumps(spec))
        (root / "TASK.md").write_text("the brief\n")
        self.args = SimpleNamespace(spec=root / "arms.json", task=root / "TASK.md", out=root / "out", per_arm=5, sample_seed=0,
                                    shuffle_seed=1, batch_chars=900)
        ea.build(self.args)
        self.set = root / "out" / "toy"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sample_and_render(self):
        key = json.loads((self.set / "key.json").read_text())
        items = json.loads((self.set / "items.json").read_text())
        self.assertEqual(len(key), 15)
        self.assertEqual({arm: sum(v["arm"] == arm for v in key.values()) for arm in ea.ARMS}, dict.fromkeys(ea.ARMS, 5))
        self.assertNotIn("prob_1", {v["example_id"] for v in key.values()})  # never submitted code
        for item_id, text in items.items():
            self.assertTrue(text.startswith(f"=== ITEM {item_id} ===\n"))
            self.assertIn("[HARNESS] code failed the visible tests", text)
            for arm in ea.ARMS:
                self.assertNotIn(arm, text)
        zero = next(i for i, v in key.items() if v["example_id"] == "prob_0") if any(v["example_id"] == "prob_0" for v in key.values()) else None
        if zero:
            self.assertIn("[HARNESS] no code submitted", items[zero])
        self.assertEqual((self.set / "TASK.md").read_text(), "the brief\n")

    def test_batches_hold_every_item_once_within_the_limit(self):
        items = json.loads((self.set / "items.json").read_text())
        seen = []
        for batch in sorted((self.set / "batches").glob("batch_*.md")):
            ids = ea.batch_items(batch)
            self.assertTrue(len(batch.read_text()) <= 900 or len(ids) == 1)
            seen += ids
        self.assertEqual(seen, list(items))  # items.json is in batch order
        self.assertEqual(sorted(seen), sorted(set(seen)))

    def test_built_set_is_not_rebuilt(self):
        with self.assertRaises(SystemExit):
            ea.build(self.args)

    def test_analysis(self):
        key = json.loads((self.set / "key.json").read_text())
        items = json.loads((self.set / "items.json").read_text())
        value = {"unsteered": "yes", "gi_v2_minus": "no", "gfar_minus": "yes"}
        for batch in sorted((self.set / "batches").glob("batch_*.md")):
            rows = []
            for n, item in enumerate(ea.batch_items(batch)):
                label = value[key[item]["arm"]]
                rows.append(dict(item=item, explanation_mismatch=label, prose_quote="Here is my reply." if label == "yes" else "",
                                 code_quote="def f(): return 1" if label == "yes" else "", note="n"))
            (self.set / "labels" / f"{batch.stem}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        key_, labels, quotes, built = ea.load_set(self.set, False)
        self.assertEqual(len(labels), 15)
        self.assertEqual((quotes["yes"], quotes["both_verbatim"], quotes["failed"]), (10, 10, []))
        stats = ea.model_stats(key_, labels, 500, False)
        self.assertEqual([stats["arms"][arm]["rate"] for arm in ea.ARMS], [100.0, 0.0, 100.0])
        self.assertEqual((stats["change_gi_v2"], stats["change_gfar"], stats["difference"]), (-100.0, 0.0, -100.0))
        self.assertEqual(stats["difference_ci"], [-100.0, -100.0])
        self.assertIn("gi_v2", items and "gi_v2")

    def test_unclear_and_not_applicable(self):
        rows = [("p1", "yes"), ("p2", "unclear"), ("p3", "not_applicable"), ("p4", "no")]
        self.assertAlmostEqual(ea.rate(rows, False), 100 / 3)
        self.assertAlmostEqual(ea.rate(rows, True), 200 / 3)

    def test_missing_labels_stop_the_analysis(self):
        with self.assertRaises(SystemExit):
            ea.load_set(self.set, False)
        self.assertEqual(ea.load_set(self.set, True)[1], {})


if __name__ == "__main__":
    unittest.main()
