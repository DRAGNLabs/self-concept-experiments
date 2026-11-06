"""main_analysis.py: each arm is read on its own seeds (unsteered on seeds 8-9 against a steered arm on seeds 0-1)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "main_analysis", ROOT / "experiments/deceiver-axis/scripts/main_analysis.py")
main_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_analysis)


def record(example_id: str, passed: bool, passed_original: bool) -> dict:
    return {"example_id": example_id, "scenario": "impossible_original", "passed": passed,
            "passed_original": passed_original, "n_attempts": 1, "stop_reason": "passed" if passed else "stopped",
            "final_code": "def f(x):\n    return x\n" if passed else None, "attempts": [{"completion": "x", "truncated": False}]}


def write_arm(root: Path, arm: str, seeds: list[int], hack_ids: set[str]) -> None:
    for seed in seeds:
        seed_dir = root / arm / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        rows = [record(f"p{i}", passed=True, passed_original=f"p{i}" not in hack_ids) for i in range(4)]
        (seed_dir / "pilot_impossible_original.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class PerArmSeeds(unittest.TestCase):
    def test_unequal_seed_sets_are_both_fully_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [8, 9], {"p0"})
            write_arm(root, "steered", [0, 1], {"p0", "p1"})
            arms = main_analysis.load_root(root, "pilot")
            result = main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"])
            self.assertEqual(result["arms"]["unsteered"]["episodes"], 8)
            self.assertEqual(result["arms"]["steered"]["episodes"], 8)
            self.assertEqual(result["arms"]["unsteered"]["seeds"], [8, 9])
            self.assertEqual(result["arms"]["steered"]["seeds"], [0, 1])
            self.assertAlmostEqual(result["arms"]["unsteered"]["rate"], 0.25)
            self.assertAlmostEqual(result["arms"]["steered"]["rate"], 0.5)
            self.assertAlmostEqual(result["arms"]["steered"]["diff_vs_unsteered"], 0.25)

    def test_judged_outcome_keys_use_each_arms_own_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [8], set())
            write_arm(root, "steered", [0], {"p0"})
            arms = main_analysis.load_root(root, "pilot")
            verdicts = {"steered/seed0/p0": "HACK"}
            result = main_analysis.analyse(arms, "judged_hack", reps=100, seed=0, verdicts=verdicts, contrasts=["steered"])
            self.assertAlmostEqual(result["arms"]["steered"]["rate"], 0.25)
            self.assertAlmostEqual(result["arms"]["unsteered"]["rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
