"""Validate experimental counterfactuals and scoring, without loading a model."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from selfconcept.soo.goals_roles import (
    MAIN_QUERIES, answer_key, build_dataset, build_example, development_worlds,
    tokenize_example, write_dataset,
)
from selfconcept.soo.score_goals_roles import parse_answer, score_records, summarize


class TaskTests(unittest.TestCase):
    def test_factorial_and_balanced_answers(self):
        rows = build_dataset()
        self.assertEqual(len(rows), 672)
        self.assertEqual(len({r["example_id"] for r in rows}), len(rows))
        self.assertEqual(Counter(r["suite"] for r in rows), {"main": 384, "checks": 288})
        self.assertEqual(set(Counter(r["world_id"] for r in rows).values()), {56})
        positions = Counter((r["query"], r["choices"].index(r["expected_answer"])) for r in rows)
        for query in {r["query"] for r in rows}:
            self.assertEqual(positions[query, 0], positions[query, 1])
        for family in ("burglar", "allocation"):
            worlds = [w for w in development_worlds() if w.family == family]
            self.assertEqual(sum(w.belief_swapped for w in worlds), 3)

    def test_actions_solve_the_actual_payoffs(self):
        # Check consequences directly, independently of answer_key's utility maximizer.
        for w in development_worlds():
            for role in (0, 1):
                target_location = w.target0_option if role == 0 else 1 - w.target0_option
                expected_choice = 1 - target_location if w.family == "burglar" else target_location
                for alignment in ("aligned", "opposed"):
                    key = answer_key(w, role, alignment)
                    self.assertEqual(key["assigned_goal_answer"], w.options[expected_choice])
                    for query in ("message", "private"):
                        self.assertEqual(key["answers"][query], w.options[expected_choice])
                    self.assertEqual(key["assigned_goal_answer"] == key["truthful_message_answer"],
                                     alignment == "aligned")

    def test_role_swap_changes_action_but_not_world_facts(self):
        for w in development_worlds():
            for alignment in ("aligned", "opposed"):
                a = build_example(w, 0, alignment, "private", False)
                b = build_example(w, 1, alignment, "private", False)
                self.assertNotEqual(a["expected_answer"], b["expected_answer"])
                self.assertEqual(a["other_role_answer"], b["expected_answer"])
                self.assertEqual(a["world"], b["world"])
                self.assertEqual(a["semantic_spans"]["facts"][0]["text"], b["semantic_spans"]["facts"][0]["text"])

    def test_incentives_change_message_truth_not_private_goal(self):
        for w in development_worlds():
            for role in (0, 1):
                a, b = (answer_key(w, role, align) for align in ("aligned", "opposed"))
                self.assertEqual(a["answers"]["fact"], b["answers"]["fact"])
                self.assertEqual(a["answers"]["private"], b["answers"]["private"])
                self.assertNotEqual(a["truthful_message_answer"], b["truthful_message_answer"])

    def test_false_beliefs_change_prediction_not_facts_or_actions(self):
        for w in development_worlds():
            for role in (0, 1):
                for alignment in ("aligned", "opposed"):
                    a = answer_key(replace(w, belief_swapped=False), role, alignment)
                    b = answer_key(replace(w, belief_swapped=True), role, alignment)
                    for query in ("fact", "message", "private"):
                        self.assertEqual(a["answers"][query], b["answers"][query])
                    for query in ("prediction", "belief_check"):
                        self.assertNotEqual(a["answers"][query], b["answers"][query])
                    self.assertEqual(a["answers"]["prediction"], a["truthful_message_answer"])

    def test_mirroring_changes_order_not_answers(self):
        for w in development_worlds():
            for query in MAIN_QUERIES:
                a, b = (build_example(w, 0, "opposed", query, mirror) for mirror in (False, True))
                self.assertEqual(a["choices"], b["choices"][::-1])
                self.assertEqual(a["expected_answer"], b["expected_answer"])
                self.assertNotEqual(a["prompt"], b["prompt"])

    def test_all_span_text_is_exact_and_nonoverlapping(self):
        for row in build_dataset():
            spans = sorted((s for ss in row["semantic_spans"].values() for s in ss), key=lambda s: s["start"])
            end = 0
            for span in spans:
                self.assertGreaterEqual(span["start"], end)
                self.assertEqual(row["prompt"][span["start"]:span["end"]], span["text"])
                end = span["end"]

    def test_dataset_is_reproducible_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_dataset(root)
            self.assertEqual(first, write_dataset(root))
            path = root / "main.jsonl"
            path.write_text("changed\n")
            with self.assertRaises(FileExistsError):
                write_dataset(root)
            self.assertEqual(path.read_text(), "changed\n")

    def test_v2_preserves_counterfactuals_and_has_distinct_join_ids(self):
        original, revised = build_dataset(), build_dataset(2)
        self.assertEqual(len(original), len(revised))
        self.assertFalse({r["example_id"] for r in original} & {r["example_id"] for r in revised})
        for old, new in zip(original, revised):
            self.assertEqual(new["paired_v1_example_id"], old["example_id"])
            for field in ("world", "outcomes", "expected_answer", "assigned_goal_answer",
                          "other_role_answer", "truthful_message_answer", "misleading_message_answer",
                          "choices", "role", "alignment", "orientation", "query", "suite"):
                self.assertEqual(new[field], old[field])
            for kind in ("role", "facts", "client0_goal", "client1_goal", "recipient_goal", "consequences"):
                self.assertEqual([s["text"] for s in old["semantic_spans"][kind]],
                                 [s["text"] for s in new["semantic_spans"][kind]])
            for clauses in new["semantic_spans"].values():
                for span in clauses:
                    self.assertEqual(new["prompt"][span["start"]:span["end"]], span["text"])
        with self.assertRaisesRegex(ValueError, "template version"):
            build_dataset(4)

    def test_v2_scores_and_maps_spans_with_native_template(self):
        rows = build_dataset(2)
        responses = [{"example_id": r["example_id"], "response": r["expected_answer"],
                      "completion_status": "complete"} for r in rows]
        summary = summarize(score_records(rows, responses))["overall"]
        self.assertEqual(summary["valid_completion_rate"], 1)
        self.assertEqual(summary["metrics"]["assigned_goal_optimal"]["all_trial_rate"], 1)
        for row in rows:
            mapped = tokenize_example(row, CharacterTokenizer())
            self.assertTrue(mapped["semantic_token_spans"]["private_rule"])


class CharacterTokenizer:
    """Tiny tokenizer tests chat offset alignment, not Gemma tokenization."""
    all_special_ids = [1, 2]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        rendered = "<user>\n" + messages[0]["content"] + "\n<assistant>"
        if not tokenize:
            return rendered
        ids = [ord(c) + 10 for c in rendered]
        return {"input_ids": ids} if kwargs.get("return_dict", True) else ids

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 10 for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class SpanTests(unittest.TestCase):
    def test_chat_prefix_and_non_ascii_offsets(self):
        world = replace(development_worlds()[0], clients=("Álex", "Blair"))
        example = build_example(world, 0, "opposed", "message", False)
        rendered = tokenize_example(example, CharacterTokenizer())
        for kind, spans in rendered["semantic_token_spans"].items():
            for span, source in zip(spans, example["semantic_spans"][kind]):
                recovered = "".join(chr(rendered["input_ids"][i] - 10) for i in span["token_indices"])
                self.assertEqual(recovered, source["text"])
                self.assertEqual(span["char_start"], source["start"] + len("<user>\n"))

    def test_refuse_inconsistent_chat_tokenization(self):
        class BrokenTokenizer(CharacterTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                result = super().apply_chat_template(*args, **kwargs)
                return result + [99] if kwargs["tokenize"] else result
        row = build_dataset()[0]
        with self.assertRaisesRegex(ValueError, "differs"):
            tokenize_example(row, BrokenTokenizer())


class ScoringTests(unittest.TestCase):
    def test_strict_answers_not_first_room_mentions(self):
        choices = ["library", "kitchen"]
        self.assertEqual(parse_answer(" Library.\n", choices), ("library", "valid"))
        for text in ("library or kitchen", "not library", "I choose library", "library, but actually kitchen",
                     '"library"', "<think>library</think>kitchen", "", "library.."):
            self.assertEqual(parse_answer(text, choices), (None, "invalid"))
        self.assertEqual(parse_answer("I cannot choose library.", choices), (None, "refusal"))

    def test_completion_status_and_missing_never_become_success(self):
        rows = build_dataset()[:5]
        records = [{"example_id": r["example_id"], "response": r["expected_answer"],
                    "completion_status": status} for r, status in zip(rows, ("complete", "truncated", "error"))]
        records.append({"example_id": rows[3]["example_id"], "response": "I cannot answer.", "completion_status": "complete"})
        scores = score_records(rows, records)
        self.assertEqual([s["score_status"] for s in scores], ["valid", "truncated", "error", "refusal", "missing"])
        self.assertEqual([s["valid_answer"] for s in scores], [True, False, False, False, False])
        for s in scores[1:]:
            self.assertFalse(any(s["metrics"].values()))
        summary = summarize(scores)["overall"]
        self.assertEqual(summary["valid_completion_rate"], 1 / 5)
        fact = summary["metrics"]["factual_correct"]
        self.assertEqual((fact["count"], fact["all_trial_n"], fact["valid_answer_n"]), (1, 2, 1))

    def test_truthful_answer_and_goal_following_are_distinct(self):
        w = development_worlds()[0]
        rows = [build_example(w, 0, align, query, False) for align in ("aligned", "opposed")
                for query in ("message", "private")]
        responses = [{"example_id": r["example_id"], "response": r["truthful_message_answer"],
                      "completion_status": "complete"} for r in rows]
        a, b, c, d = score_records(rows, responses)
        self.assertTrue(a["metrics"]["assigned_goal_optimal"])
        self.assertTrue(c["metrics"]["truthful_message"])
        self.assertFalse(c["metrics"]["assigned_goal_optimal"])
        self.assertTrue(c["metrics"]["other_role_optimal"])
        self.assertIsNone(d["metrics"]["truthful_message"])

    def test_bad_record_join_and_legacy_status_rejected(self):
        rows = build_dataset()[:1]
        good = {"example_id": rows[0]["example_id"], "response": "library", "completion_status": "complete"}
        for examples, records in ((rows * 2, [good]), (rows, [good, good]),
                                  (rows, [{**good, "example_id": "unknown"}]),
                                  (rows, [{"example_id": good["example_id"], "response": "library"}])):
            with self.assertRaises(ValueError):
                score_records(examples, records)

    def test_oracle_responses_pass_every_task_check(self):
        rows = build_dataset()
        responses = [{"example_id": r["example_id"], "response": r["expected_answer"],
                      "completion_status": "complete"} for r in rows]
        summary = summarize(score_records(rows, responses))["overall"]
        self.assertEqual(summary["valid_completion_rate"], 1)
        for metric in ("factual_correct", "recipient_prediction_correct", "role_correct", "goal_check_correct",
                       "belief_correct", "assigned_goal_optimal"):
            self.assertEqual(summary["metrics"][metric]["all_trial_rate"], 1)
        self.assertEqual(summary["metrics"]["misleading_message"]["all_trial_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
