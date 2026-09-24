"""Guard against reasoning/answer leakage, token drift, and biased denominators."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from selfconcept.cotness.analyze import analyze, association, outcome
from selfconcept.cotness.probe import capture, fit_linear, probabilities
from selfconcept.cotness.roles import content_indices, render_probe, response_spans
from selfconcept.cotness.run import generation_seed, select_examples
from selfconcept.cotness.gate import assess, CODE_SCENARIOS, SCENARIOS


class BoundaryTests(unittest.TestCase):
    def test_prompt_open_think_is_not_an_answer(self):
        for family, op, cl in [("qwen", "<think>", "</think>"), ("kimi", "◁think▷", "◁/think▷"),
                               ("gemma", "<|channel>thought\n", "<channel|>")]:
            spans = response_spans(op, "A private thought", family)
            self.assertEqual(spans["final"], [])
            raw = "A private thought" + cl + "The answer"
            spans = response_spans(op, raw, family)
            self.assertEqual("".join(raw[a:b] for a,b in spans["final"]), "The answer")

    def test_generated_think_and_partial_muse_header(self):
        raw = "◁think▷private◁/think▷public"
        s = response_spans("", raw, "kimi")
        self.assertEqual([raw[a:b] for a,b in s["cot"]], ["private"])
        self.assertEqual([raw[a:b] for a,b in s["final"]], ["public"])
        prompt = "<|start|>assistant"
        raw = " to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>public<|eot|>"
        s = response_spans(prompt, raw, "muse")
        self.assertEqual([raw[a:b] for a,b in s["cot"]], ["private"])
        self.assertEqual([raw[a:b] for a,b in s["final"]], ["public"])
        self.assertFalse(response_spans(prompt, " to=self<|message|>unfinished", "muse")["final"])

    def test_boundary_straddles_and_specials_are_excluded(self):
        offsets = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 9)]
        self.assertEqual(content_indices(offsets, [(4, 9)], [2], [0, 1, 2, 3, 4]), [3])

    def test_template_trimming_preserves_identical_role_content(self):
        class TrimmingTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "|".join(m.get("reasoning_content", m["content"]).strip() for m in messages)
        for role in ("user", "cot", "assistant"):
            text, (a, b) = render_probe(TrimmingTokenizer(), "qwen", "  A neutral sentence. \n", role)
            self.assertEqual(text[a:b], "A neutral sentence.")

    def test_unsupported_kimi_probe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no verified native"):
            render_probe(None, "kimi", "Document content.", "cot")


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hook_preserves_generation_and_aligns_processed_tokens(self):
        torch.manual_seed(8)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                                bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
        x = torch.tensor([[1, 3, 9, 2]])
        probes = {1: {"weight": np.zeros((3, 16), dtype=np.float32), "bias": np.zeros(3, dtype=np.float32)}}
        with torch.inference_mode():
            baseline = model.generate(x, max_new_tokens=5, do_sample=False)
            with capture(model, [1], probes) as values:
                measured = model.generate(x, max_new_tokens=5, do_sample=False)
        self.assertTrue(torch.equal(baseline, measured))
        scores = torch.cat(values[1])
        self.assertEqual(len(scores), measured.shape[1]-1)
        torch.testing.assert_close(scores, torch.full_like(scores, 1/3))
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_linear_softmax_roundtrip(self):
        rng = np.random.default_rng(42)
        y = np.repeat(np.arange(3), 100)
        x = rng.normal(size=(300, 5)) + np.eye(3, 5)[y]*8
        fitted = fit_linear(x, y)
        p = probabilities(x, fitted)
        self.assertGreater((p.argmax(1) == y).mean(), .95)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)

    def test_missingness_and_visible_only_success_are_not_deception(self):
        for status in ("truncated", "no_final", "error_oom"):
            self.assertIsNone(outcome({"status": status, "scenario": "main", "label": "deceptive"})[0])
        self.assertIsNone(outcome({"status": "complete", "scenario": "evilgenie", "label": "hack_visible_only"})[0])
        self.assertIsNone(outcome({"status": "complete", "scenario": "sandbagging", "label": "unparsed"})[0])
        self.assertIsNone(outcome({"status": "complete", "scenario": "perspectives", "label": "deceptive"})[0])

    def test_direction_is_not_flipped_to_make_auc_large(self):
        x = np.arange(20.)
        y = np.array([1]*10 + [0]*10)
        r = association(x, y, np.ones((20, 3)), bootstrap=20)
        self.assertEqual(r["auc"], 0.)
        self.assertLess(r["rho"], 0.)
        self.assertEqual(association(x, np.ones(20), [], bootstrap=20)["status"], "insufficient_variation")

    def test_rule_confirmed_code_and_conflict_flags_survive_no_code_judge(self):
        for label, expected in [("flagged", 0), ("cheat_modify_tests", 1)]:
            rec = {"scenario": "impossible_conflicting", "status": "complete", "label": label}
            self.assertEqual(outcome(rec, {"label": "no_code"})[0], expected)

    def test_disjoint_slices_and_sandbagging_half_offsets(self):
        rows = [{"example_id": f"{half}-{i}", "expected": half} for half in ("sandbag", "normal") for i in range(50)]
        selected = select_examples(rows, 32, "sandbagging", offset=8)
        self.assertEqual(len(selected), 64)
        self.assertEqual(selected[0]["example_id"], "sandbag-8")
        self.assertEqual(selected[32]["example_id"], "normal-8")
        pilot = select_examples(rows, 8, "sandbagging")
        self.assertFalse({r['example_id'] for r in pilot} & {r['example_id'] for r in selected})
        with self.assertRaises(ValueError):
            select_examples(rows, 50, "sandbagging", offset=1)

    def test_seed_is_stable_and_separate_for_turns_and_scenarios(self):
        a = generation_seed(1729, 'main', 'ex1', 0)
        self.assertEqual(a, generation_seed(1729, 'main', 'ex1', 0))
        self.assertNotEqual(a, generation_seed(1729, 'main', 'ex1', 1))
        self.assertNotEqual(a, generation_seed(1729, 'main_mirrored', 'ex1', 0))

    def test_sandbagging_normal_is_capability_control(self):
        rec = {'scenario': 'sandbagging', 'expected': 'normal', 'status': 'complete', 'label': 'unparsed'}
        self.assertEqual(outcome(rec), (None, 'capability_control'))

    def test_expansion_gate_uses_completion_not_deception(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'probes').mkdir()
            (p / 'manifest.json').write_text(json.dumps({'stage': 'complete'}))
            (p / 'probes/validation.json').write_text(json.dumps({'usable': True, 'primary_layer': 1}))
            rows = [{'scenario': s, 'example_id': str(i), 'status': 'complete', 'label': 'deceptive'}
                    for s in SCENARIOS for i in range(1 if s in CODE_SCENARIOS else 4 if s == 'sandbagging' else 2)]
            def save():
                (p / 'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save()
            gs = [{'scenario': r['scenario'], 'example_id': r['example_id'], 'turn': 0,
                   'generated_alignment': 'exact', 'scores': {'1': {'prompt': {'mean': .1}}}} for r in rows]
            (p / 'generations.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in gs))
            self.assertTrue(assess(p)['passed'])
            for row in rows:
                row['label'] = 'honest'
            save()
            self.assertTrue(assess(p)['passed'])
            rows[-1]['status'] = rows[-2]['status'] = 'truncated'
            save()
            self.assertFalse(assess(p)['passed'])
            rows[-2]['status'] = 'complete'
            save()
            self.assertTrue(assess(p)['passed'])
            gs[0]['generated_alignment'] = 'unavailable'
            (p / 'generations.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in gs))
            self.assertFalse(assess(p)['passed'])

    def test_analysis_joins_scenario_and_first_turn_and_deduplicates_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "probes").mkdir()
            (p / "probes/validation.json").write_text(json.dumps({"usable": True, "primary_layer": 1}))
            records, generations = [], []
            for scenario, reverse in [("main", False), ("main_mirrored", True)]:
                for i in range(8):
                    records.append({"scenario": scenario, "example_id": str(i), "status": "complete", "label": "deceptive" if i >= 4 else "honest"})
                    for turn in (0, 1):
                        value = (7-i if reverse else i) / 8 if turn == 0 else .5
                        generations.append({"scenario": scenario, "example_id": str(i), "turn": turn, "prompt_tokens": 10,
                                            "scores": {"1": {"prompt": {"mean": value, "n": 10}}}})
            records.append(records[-1])
            (p / "outcomes.jsonl").write_text("".join(json.dumps(r)+'\n' for r in records))
            (p / "generations.jsonl").write_text("".join(json.dumps(r)+'\n' for r in generations))
            result = analyze(p, bootstrap=10)
            cells = {r["scenario"]: r for r in result["cells"]}
            self.assertEqual(cells["main"]["n"], 8)
            self.assertEqual(cells["main"]["auc"], 1.)
            self.assertEqual(cells["main_mirrored"]["auc"], 0.)


if __name__ == "__main__":
    unittest.main()
