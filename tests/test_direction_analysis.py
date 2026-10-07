import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from selfconcept.correlation.direction_analysis import held_out_diff_means, main as analyze_main, orthogonalize, partial_rank_correlations


class DirectionAnalysisTests(unittest.TestCase):
    def test_orthogonalize_removes_reference_component(self):
        reference = np.array([1., 0, 0, 0])
        direction = np.array([.6, .8, 0, 0])
        result = orthogonalize(direction, reference)
        self.assertAlmostEqual(result @ reference, 0)
        self.assertAlmostEqual(np.linalg.norm(result), 1)

    def test_partial_correlation_removes_control(self):
        rng = np.random.default_rng(0)
        control = rng.normal(size=200)
        labels = (control + rng.normal(scale=.1, size=200) > 0).astype(int)
        scores = np.column_stack([control + rng.normal(scale=.1, size=200), rng.normal(size=200)])
        raw = partial_rank_correlations(scores, labels, rng.normal(size=(200, 1)), ["s"] * 200)
        adjusted = partial_rank_correlations(scores, labels, control[:, None], ["s"] * 200)
        self.assertGreater(raw[0], .8)
        self.assertLess(abs(adjusted[0]), .3)

    def test_held_out_diff_means_excludes_own_problem(self):
        residuals = np.array([[1., 0], [0, 0], [1, 0], [0, 0]])
        labels = np.array([1, 0, 1, 0])
        scores = held_out_diff_means(residuals, labels, ["a", "a", "b", "b"])
        self.assertTrue(np.all(scores[labels == 1] > scores[labels == 0]))
        self.assertTrue(np.isnan(held_out_diff_means(residuals[:2], labels[:2], ["a", "a"])).all())

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / "shard_0"
            shard.mkdir()
            identity = {"model": "tiny", "shards": 1, "shard": 0, "layers": [1], "limit": None, "n_expected": 8}
            (shard / "manifest.json").write_text(json.dumps({"identity": identity, "stage": "complete", "num_layers": 2, "hidden_size": 4}))
            rows = []
            for i in range(8):
                means = np.zeros((3, 4), dtype=np.float32)
                means[:, 0] = 10 * (i % 2)  # signal well above the unit noise so held-out fits recover it
                means[:, 1:] = np.random.default_rng(i).normal(size=(3, 3))
                np.savez(shard / f"{i}.npz", layer_1=means)
                outcome = {"status": "complete", "scenario": "test", "label": "x", "verdict": "HACK" if i % 2 else "unflagged",
                           "problem": str(i // 2), "stratum": str(i // 2)}
                if i % 4 == 1:
                    outcome["strength"] = "STRONG"
                rows.append({"example_id": str(i), "scenario": "test", "status": "complete", "residuals": f"{i}.npz",
                             "region_counts": [2, 2, 2], "prompt_tokens": 10, "outcome": outcome})
            (shard / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            good, reference = torch.zeros(2, 4), torch.zeros(2, 4)
            good[:, 0], reference[:, 1] = 1, 1
            torch.save({"axis": good, "model": "tiny"}, root / "good.pt")
            torch.save(reference, root / "reference.pt")
            argv = ["analyze", str(shard), "--direction", f"good={root / 'good.pt'}", "--reference", f"aa={root / 'reference.pt'}",
                    "--out", str(root / "analysis"), "--random-count", "8", "--bootstrap", "10"]
            with patch.object(sys, "argv", argv):
                analyze_main()
            result = json.loads((root / "analysis/analysis.json").read_text())
            primary = next(c for c in result["cells"] if c["primary"])
            self.assertEqual(primary["region"], "cot")
            self.assertEqual(primary["directions"]["good"]["auroc"]["value"], 1)
            self.assertEqual(primary["directions"]["good"]["strong_only_auroc"], 1)
            self.assertEqual(primary["directions"]["good|orth_aa"]["auroc"]["value"], 1)
            self.assertIn("aa", primary["directions"])  # noise dimension: any AUROC
            self.assertEqual(primary["held_out_diff_means_auroc"], 1)
            self.assertAlmostEqual(result["geometry"]["1"]["good"]["cosine_with_aa"], 0)
            self.assertTrue((root / "analysis/analysis.md").exists())


if __name__ == "__main__":
    unittest.main()
