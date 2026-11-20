"""Tests for the sycophancy readout helpers (synthetic data, no model)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "sycophancy_readout",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "deceiver-axis", "scripts", "sycophancy_readout.py"),
)
sr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sr)


class AurocTest(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(sr.auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(sr.auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)
        self.assertEqual(sr.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5)

    def test_bootstrap_over_questions(self):
        pos = {q: np.array([2.0, 3.0]) for q in range(10)}
        neg = {q: np.array([0.0, 1.0]) for q in range(10)}
        out = sr.boot(pos, neg, 50, np.random.default_rng(0))
        self.assertEqual((out["auroc"], out["ci"], out["n_pos"], out["n_neg"]), (1.0, [1.0, 1.0], 20, 20))
        self.assertEqual(sr.question("pos_p4_q98"), 98)


class ProbesTest(unittest.TestCase):
    def test_counts_per_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            blind = Path(tmp)
            (blind / "labels").mkdir()
            key, rows = {}, []
            for n, (setting, subset, label) in enumerate([("unsteered", "probes_low", "no"), ("gi_p0.4", "probes_low", "yes"),
                                                          ("gi_p0.4", "probes_high", "unclear"), ("gi_p0.4", "probes_striking", "yes")]):
                key[f"id{n}"] = dict(set=subset, setting=setting, seed=0, q=1, question="Q?", truncated=False, empty_final=n == 2, chars=10)
                rows.append(dict(file=f"id{n}.md", final_deception=label, scheming_voice="no", refusal="no", quote="q", note="n"))
            (blind / "key.json").write_text(json.dumps(key))
            (blind / "labels" / "batch_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (blind / "labels" / "usage.jsonl").write_text('{"cost": 1}\n')
            out = blind / "out.json"
            sr.probes(SimpleNamespace(blind=blind, questions=["Q?"], sets=["probes_low", "probes_high"],
                                      labels=["final_deception", "scheming_voice", "refusal"], out=out))
            table = json.loads(out.read_text())["Q?"]
            self.assertEqual((table["unsteered"]["n"], table["unsteered"]["final_deception"]), (1, 0))
            cell = table["gi_p0.4"]
            self.assertEqual((cell["n"], cell["final_deception"], cell["unclear"], cell["cut_or_empty"]), (2, 1, 1, 1))
            self.assertEqual([y["item"] for y in cell["yes_items"]], ["id1"])


if __name__ == "__main__":
    unittest.main()
