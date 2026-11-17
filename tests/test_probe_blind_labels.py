"""Tests for the blind probe-label bundle and its analysis (synthetic answers, no model, no CLI call)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "probe_blind_labels",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "deceiver-axis", "scripts", "probe_blind_labels.py"),
)
pbl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pbl)

PROMPTS = {"low": ["q one", "q two", "q three"], "striking": ["q two", "role prompt"]}


def _write(root: Path, setting: str, truncated: bool = False) -> None:
    for set_name, prompts in PROMPTS.items():
        for seed in (0, 1):
            path = root / set_name / setting / f"seed{seed}"
            path.mkdir(parents=True)
            rows = [{"question": q, "reasoning": "thinking", "final": f"{setting} answer {i}", "truncated": truncated and set_name == "low",
                     "steering": {"mode": "add"}} for i, q in enumerate(prompts)]
            (path / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class ProbeBlindLabelsTest(unittest.TestCase):
    def test_build_usability_and_contrast(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "src", Path(d) / "out"
            for setting in ("unsteered", "gi_p0.4", "gf_p0.4", "gf_m0.4"):
                _write(src, setting, truncated=setting == "gf_m0.4")
            args = SimpleNamespace(out=str(out), sets=["low", "striking"], seeds=[0, 1], seed=1, batch_chars=600, batch_answers=24,
                                   arm=[f"{s}={src}" for s in ("unsteered", "gi_p0.4", "gf_p0.4", "gf_m0.4")], asks_deception=["striking:2"])
            pbl.build(args)
            key = json.loads((out / "key.json").read_text())
            self.assertEqual(len(key), 4 * 2 * 5)
            listed = [line for p in sorted((out / "batches").glob("batch_*.list")) for line in p.read_text().split()]
            self.assertEqual(sorted(Path(n).stem for n in listed), sorted(key))
            for answer_id, record in key.items():
                text = (out / "dossiers" / f"{answer_id}.md").read_text()
                self.assertNotIn(record["setting"].split("_")[0] + "_", text.replace(f"{record['setting']} answer", ""))
                self.assertNotIn("steering", text)
                self.assertEqual("Note on label 1" in text, record["question"] == "role prompt")
            with self.assertRaises(SystemExit):
                pbl.build(args)
            usable = pbl.usability(key)
            self.assertTrue(usable["gf_m0.4"]["low"]["degraded"])
            self.assertFalse(usable["gf_m0.4"]["striking"]["degraded"])
            self.assertFalse(usable["gi_p0.4"]["low"]["degraded"])
            # gi_p0.4 answers "yes" on every prompt, the others never
            values = {s: {} for s in ("unsteered", "gi_p0.4")}
            for record in key.values():
                if record["setting"] in values:
                    values[record["setting"]][(record["set"], record["q"], record["seed"])] = (float(record["setting"] == "gi_p0.4"), record["question"])
            row = pbl.contrast(values, "gi_p0.4", "unsteered", ["low", "striking"], 2000, np.random.default_rng(0))
            self.assertEqual((row["n"], row["prompts"], row["delta"], row["ci"]), (10, 4, 1.0, [1.0, 1.0]))
            self.assertLess(row["p"], 0.2)  # four prompts: 2 of 16 sign patterns reach the observed sum
            self.assertEqual(pbl.contrast(values, "gi_p0.4", "unsteered", ["low"], 200, np.random.default_rng(0))["n"], 6)

    def test_holm(self):
        self.assertEqual(pbl.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})


if __name__ == "__main__":
    unittest.main()
