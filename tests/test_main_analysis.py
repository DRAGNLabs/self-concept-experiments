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

    def test_honest_hidden_pass_reads_each_arm_on_its_own_seeds(self):
        """2026-11-09: the honest hidden pass read every arm on the reference's seeds, so a steered arm on disjoint seeds had
        no episodes (NaN rate) and one overlapping on a few seeds was read on those alone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [8, 9], {"p0"})
            write_arm(root, "steered", [0, 1], {"p0", "p1"})
            arms = main_analysis.load_root(root, "pilot")
            result = main_analysis.analyse_honest_hidden_pass(arms, reps=200, seed=0, contrasts=["steered"])
            self.assertEqual(result["arms"]["unsteered"]["episodes"], 8)
            self.assertEqual(result["arms"]["steered"]["episodes"], 8)
            # every honest (non-proxy-hack) episode passes the hidden suite in write_arm, so both rates are 1.0
            self.assertAlmostEqual(result["arms"]["unsteered"]["rate"], 1.0)
            self.assertAlmostEqual(result["arms"]["steered"]["rate"], 1.0)
            self.assertAlmostEqual(result["arms"]["steered"]["diff_vs_unsteered"], 0.0)


if __name__ == "__main__":
    unittest.main()


class Holm(unittest.TestCase):
    def test_known_vector_is_step_down_monotone_and_capped(self):
        adjusted = main_analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.6})
        self.assertAlmostEqual(adjusted["a"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.09)
        self.assertAlmostEqual(adjusted["b"], 0.09)  # 2 * 0.04 = 0.08 is raised to the running maximum
        self.assertAlmostEqual(adjusted["d"], 0.6)
        self.assertEqual(main_analysis.holm({"a": 0.9, "b": 0.8})["b"], 1.0)


class ExactSignFlip(unittest.TestCase):
    def test_matches_full_enumeration(self):
        import itertools
        import numpy as np
        diff = np.array([3, -1, 0, 2, 2, -4, 1]) / 12
        observed = abs(diff.sum())
        sums = [abs(sum(sign * d for sign, d in zip(signs, diff))) for signs in itertools.product([-1, 1], repeat=len(diff))]
        enumerated = sum(s >= observed - 1e-12 for s in sums) / len(sums)
        self.assertAlmostEqual(main_analysis.sign_flip_p_exact(diff, 12), enumerated)

    def test_goes_below_the_resampling_floor_and_handles_edges(self):
        import numpy as np
        self.assertAlmostEqual(main_analysis.sign_flip_p_exact(np.full(30, 2 / 12), 12), 2 / 2 ** 30)
        self.assertEqual(main_analysis.sign_flip_p_exact(np.zeros(5), 12), 1.0)
        self.assertIsNone(main_analysis.sign_flip_p_exact(np.array([0.1234, 0.2]), 12))  # not on the grid
        self.assertIsNone(main_analysis.sign_flip_p_exact(np.array([1 / 12, np.nan]), 12))

    def test_analyse_reports_the_exact_p(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [0, 1], set())
            write_arm(root, "steered", [0, 1], {"p0", "p1", "p2", "p3"})
            arms = main_analysis.load_root(root, "pilot")
            result = main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"])
            self.assertEqual(result["contrasts"]["steered"]["p_method"], "exact")
            self.assertAlmostEqual(result["contrasts"]["steered"]["p_raw"], 2 / 2 ** 4)  # four problems, all up
            self.assertGreaterEqual(result["arms"]["steered"]["p_signflip_resampled"], 1 / 201)


class Completeness(unittest.TestCase):
    def test_arm_without_a_problem_is_an_error_not_the_smallest_p(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [0, 1], {"p0"})
            write_arm(root, "steered", [0, 1], {"p0"})
            for seed in (0, 1):  # drop problem p3 from every seed of the steered arm
                path = root / "steered" / f"seed{seed}" / "pilot_impossible_original.jsonl"
                path.write_text("".join(line + "\n" for line in path.read_text().splitlines() if '"p3"' not in line))
            arms = main_analysis.load_root(root, "pilot")
            with self.assertRaisesRegex(ValueError, "no episode for 1 of 4 problems"):
                main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"])
            with self.assertRaisesRegex(ValueError, "no episode"):
                main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"], allow_incomplete=True)

    def test_partly_written_contrast_arm_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_arm(root, "unsteered", [0, 1], {"p0"})
            write_arm(root, "steered", [0, 1], {"p0"})
            path = root / "steered" / "seed1" / "pilot_impossible_original.jsonl"
            path.write_text("".join(line + "\n" for line in path.read_text().splitlines()[:2]))
            arms = main_analysis.load_root(root, "pilot")
            with self.assertRaisesRegex(ValueError, "lacks 2 of 8 episodes"):
                main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"])
            result = main_analysis.analyse(arms, "hack_proxy", reps=200, seed=0, contrasts=["steered"], allow_incomplete=True)
            self.assertEqual(result["arms"]["steered"]["episodes"], 6)
