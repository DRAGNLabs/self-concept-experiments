"""cap_size_match.py: the matched threshold reproduces the reference cap's open-loop size."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "cap_size_match", Path(__file__).resolve().parents[1] / "experiments/deceiver-axis/scripts/cap_size_match.py")
csm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(csm)
KEYS = [str(round(0.05 * i, 2)) for i in range(1, 20)]


def sidecar(path: Path, mean: float, sd: float, turns: int = 3) -> None:
    z = np.array([-1.645, -1.282, -1.036, -0.842, -0.674, -0.524, -0.385, -0.253, -0.126, 0.0,
                  0.126, 0.253, 0.385, 0.524, 0.674, 0.842, 1.036, 1.282, 1.645])
    with path.open("w") as f:
        for t in range(turns):
            q = dict(zip(KEYS, (mean + 0.1 * t + sd * z).tolist()))
            f.write(json.dumps({"layers": {"5": {"generated_projection": {"n": 10, "quantiles": q}}}}) + "\n")
        f.write(json.dumps({"layers": {"5": {"generated_projection": {"n": 0, "quantiles": {}}}}}) + "\n")


class CapSizeMatch(unittest.TestCase):
    def test_same_distribution_returns_reference_tau(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            sidecar(p, mean=-100.0, sd=40.0)
            turns = csm.turn_quantiles(p, 5)
            self.assertEqual(len(turns), 3)
            size = csm.cap_size(turns, -100.0)
            self.assertAlmostEqual(csm.match_tau(turns, size), -100.0, places=4)

    def test_narrow_direction_needs_a_threshold_below_its_distribution(self):
        with tempfile.TemporaryDirectory() as d:
            wide, narrow = Path(d) / "w.jsonl", Path(d) / "n.jsonl"
            sidecar(wide, mean=-100.0, sd=40.0)
            sidecar(narrow, mean=3.0, sd=4.0)
            size = csm.cap_size(csm.turn_quantiles(wide, 5), -100.0)
            turns = csm.turn_quantiles(narrow, 5)
            self.assertLess(csm.cap_size(turns, 3.1) / size, 0.2)  # the same-percentile cap is far gentler
            tau = csm.match_tau(turns, size)
            self.assertAlmostEqual(csm.cap_size(turns, tau), size, places=5)
            self.assertEqual(csm.share_above(turns, tau), 1.0)  # every token is moved
            self.assertAlmostEqual(tau, 3.1 - size, places=4)   # linear regime: mean - size


if __name__ == "__main__":
    unittest.main()
