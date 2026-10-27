"""Check generation-stop and resume safeguards without loading Gemma."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from selfconcept.soo.goals_roles import build_dataset
from selfconcept.soo.run_goals_roles import append_record, baseline_gate, check_resume, completion
from selfconcept.soo.score_goals_roles import score_records


class CompletionTests(unittest.TestCase):
    def test_all_model_eos_ids_and_batch_padding(self):
        for eos in (1, 106, 50):
            self.assertEqual(completion([233, eos, 0, 0], [1, 106, 50], 32), ([233, eos], "complete"))

    def test_answer_at_budget_is_truncated_until_eos(self):
        self.assertEqual(completion([233, 222], [1], 2), ([233, 222], "truncated"))
        self.assertEqual(completion([233, 1], [1], 2), ([233, 1], "complete"))

    def test_early_stop_without_eos_rejected(self):
        with self.assertRaises(ValueError):
            completion([233], [1], 32)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_dataset()[:2]
        self.manifest = {"revision": "abc", "batch_size": 8, "artifact_sha256": {"data": "123"}}
        self.record = {"example_id": self.rows[0]["example_id"], "response": self.rows[0]["expected_answer"],
                       "completion_status": "complete"}

    def test_partial_resume_matches_ids(self):
        self.assertEqual(check_resume(self.manifest, deepcopy(self.manifest), [self.record], self.rows),
                         {self.record["example_id"]})

    def test_configuration_change_rejected(self):
        saved = {**self.manifest, "batch_size": 4}
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            check_resume(self.manifest, saved, [self.record], self.rows)

    def test_duplicate_records_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_resume(self.manifest, self.manifest, [self.record, self.record], self.rows)

    def test_append_is_readable_before_file_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            with path.open("a") as handle:
                append_record(handle, self.record)
                self.assertEqual(json.loads(path.read_text()), self.record)


class BaselineGateTests(unittest.TestCase):
    thresholds = {"valid_answers_each_family": .95,
                  "fact_prediction_private_and_comprehension_each_family": .90,
                  "opposed_message_assigned_goal_each_family": .75}

    def test_perfect_oracle_passes_but_missing_data_cannot(self):
        rows = build_dataset()
        responses = [{"example_id": r["example_id"], "response": r["expected_answer"],
                      "completion_status": "complete"} for r in rows]
        scores = score_records(rows, responses)
        self.assertTrue(baseline_gate(scores, self.thresholds)["passed"])
        self.assertFalse(baseline_gate([], self.thresholds)["passed"])
        self.assertFalse(baseline_gate(score_records(rows, []), self.thresholds)["passed"])

    def test_good_pooled_score_cannot_hide_failed_private_actions(self):
        rows = build_dataset()
        responses = [{"example_id": r["example_id"],
                      "response": r["other_role_answer"] if r["query"] == "private" and r["family"] == "burglar"
                                  else r["expected_answer"], "completion_status": "complete"} for r in rows]
        result = baseline_gate(score_records(rows, responses), self.thresholds)
        self.assertFalse(result["passed"])
        failed = [r for r in result["checks"] if not r["passed"]]
        self.assertEqual([(r["family"], r["check"]) for r in failed], [("burglar", "private")])


if __name__ == "__main__":
    unittest.main()
