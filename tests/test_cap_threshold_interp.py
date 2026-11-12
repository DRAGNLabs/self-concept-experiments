import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "deceiver-axis" / "scripts"))
from cap_threshold_interp import interpolate  # noqa: E402

QS = (.05, .25, .5, .75, .95)


def row(turn: int, rng: np.random.Generator, layers: dict[int, tuple[float, float]]) -> dict:
    out = {}
    for layer, (mu, sigma) in layers.items():
        values = rng.normal(mu, sigma, 4000)
        out[str(layer)] = {"generated_projection": {"n": len(values), "mean": float(values.mean()),
                                                    "quantiles": {str(q): float(np.quantile(values, q)) for q in QS}}}
    return {"example_id": f"e{turn}", "turn": turn, "layers": out}


class InterpolateTest(unittest.TestCase):
    def test_p35_lands_near_the_gaussian_p35_and_between_the_neighbours(self):
        rng = np.random.default_rng(0)
        layers = {20: (100.0, 30.0), 30: (10.0, 5.0), 38: (-60.0, 20.0)}
        rows = [row(t, rng, layers) for t in range(6)]
        tau, note = interpolate(rows, 0.35, None, linear=False)
        self.assertEqual(note["method"]["stored_neighbours"], [0.25, 0.5])
        self.assertAlmostEqual(note["method"]["weight"], 0.4288, places=3)
        for layer, (mu, sigma) in layers.items():
            truth = mu + sigma * statistics.NormalDist().inv_cdf(0.35)
            self.assertAlmostEqual(tau[str(layer)], truth, delta=0.05 * sigma)
            n = note["layers"][str(layer)]
            self.assertLess(n["median_p25"], tau[str(layer)])
            self.assertLess(tau[str(layer)], n["median_p50"])

    def test_stored_quantile_and_layer_filter(self):
        rng = np.random.default_rng(1)
        rows = [row(t, rng, {20: (0.0, 1.0), 21: (5.0, 1.0)}) for t in range(3)]
        tau, note = interpolate(rows, 0.25, [21], linear=False)
        self.assertEqual(list(tau), ["21"])
        self.assertEqual(note["method"]["weight"], 0.0)
        self.assertEqual(tau["21"], float(np.median([r["layers"]["21"]["generated_projection"]["quantiles"]["0.25"] for r in rows])))

    def test_turns_without_tokens_are_skipped(self):
        rng = np.random.default_rng(2)
        rows = [row(t, rng, {20: (0.0, 1.0)}) for t in range(3)]
        rows.append({"example_id": "empty", "turn": 0, "layers": {"20": {"generated_projection": {"n": 0}}}})
        tau, note = interpolate(rows, 0.35, None, linear=True)
        self.assertEqual(note["layers"]["20"]["turns"], 3)
        self.assertAlmostEqual(note["method"]["weight"], 0.4)
        self.assertTrue(np.isfinite(tau["20"]))


if __name__ == "__main__":
    unittest.main()
