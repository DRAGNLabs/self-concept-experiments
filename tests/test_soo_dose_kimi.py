"""Software checks for the dose/position round, the Qwen band-steering round and the Kimi adapter round.

Run: .venv-313/bin/python -m unittest discover -s tests -p 'test_soo_dose_kimi.py' -v
Tiny random models on CPU; no model download. These check the code paths the rounds add
(content-token steering, fused-projection slices as band offsets, DeltaNet module lookup,
post-think classification, batched generation), not any experimental result.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.evaluate import answer_text, content_span
from selfconcept.soo.steering import (
    PositionalSteering, apply_module_steering, attn_proj, band_supports, get_band_offsets, random_matched_offsets,
    response_marker,
)
from test_soo_band_steer import capture_proj, load_script, save_tiny_model_with_tokenizer
from test_soo_hybrid_subset import lora as hybrid_lora, tiny_hybrid
from test_soo_lora_subset import tiny_model


class ContentPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = tiny_model()
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 6, 7], [0, 1, 6, 9, 4, 8]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])}
        self.offsets = {"1:v_proj": torch.full((16,), 0.5)}

    def test_offset_reaches_only_the_content_span(self):
        base, _ = capture_proj(self.model, 1, "v_proj", self.batch)
        positions = PositionalSteering(None, "content")
        with apply_module_steering(self.model, self.offsets, 1.0, positions):
            positions.set_spans([(1, 4), (2, 5)])
            steered, _ = capture_proj(self.model, 1, "v_proj", self.batch)
        moved = (steered - base).abs().sum(-1) > 1e-6
        self.assertEqual(moved.tolist(), [[False, True, True, True, False, False], [False, False, True, True, True, False]])

    def test_spans_must_be_set_for_every_prompt(self):
        positions = PositionalSteering(None, "content")
        with apply_module_steering(self.model, self.offsets, 1.0, positions):
            positions.set_spans([(1, 4), (2, 5)])
            capture_proj(self.model, 1, "v_proj", self.batch)
            with self.assertRaises(RuntimeError):
                capture_proj(self.model, 1, "v_proj", self.batch)

    def test_generated_tokens_are_never_steered(self):
        positions = PositionalSteering(None, "content")
        seen = []
        with apply_module_steering(self.model, self.offsets, 1.0, positions):
            v = attn_proj(get_decoder_layers(self.model)[1], "v_proj")
            h = v.register_forward_hook(lambda _m, _i, o: seen.append(positions.mask(o)[..., 0].clone()))
            positions.set_spans([(0, 3)])
            with torch.no_grad():
                self.model.generate(input_ids=torch.tensor([[1, 4, 5, 9]]), max_new_tokens=3, do_sample=False, pad_token_id=0)
            h.remove()
        self.assertEqual(seen[0].tolist(), [[1.0, 1.0, 1.0, 0.0]])
        self.assertTrue(all(m.sum() == 0 for m in seen[1:]))


class NoGenerationPromptTests(unittest.TestCase):
    """Mistral's template adds nothing for the assistant turn: 'prompt' is every prefill token, 'response' the generated ones."""

    def setUp(self):
        self.model = tiny_model()
        self.offsets = {"1:v_proj": torch.full((16,), 0.5)}

    def test_response_marker_is_empty_when_the_template_adds_nothing(self):
        class Tok:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
                return {"input_ids": [1, 2, 3]}
        self.assertEqual(response_marker(Tok()), [])

    def test_empty_marker_splits_at_the_first_generated_token(self):
        seen = {}
        for mode in ("prompt", "response"):
            positions = PositionalSteering([], mode)
            masks = []
            with apply_module_steering(self.model, self.offsets, 1.0, positions):
                v = attn_proj(get_decoder_layers(self.model)[1], "v_proj")
                h = v.register_forward_hook(lambda _m, _i, o: masks.append(positions.mask(o)[..., 0].clone()))
                with torch.no_grad():
                    self.model.generate(input_ids=torch.tensor([[1, 4, 5, 9]]), max_new_tokens=3, do_sample=False, pad_token_id=0)
                h.remove()
            seen[mode] = [m.sum().item() for m in masks]
        self.assertEqual(seen["prompt"], [4.0, 0.0, 0.0])
        self.assertEqual(seen["response"], [0.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            PositionalSteering(None, "prompt")


class SliceOffsetTests(unittest.TestCase):
    def test_slices_zero_the_other_channels_and_confine_randoms(self):
        data = {"offsets": {"all": {"3:in_proj_qkv": torch.arange(1., 11.), "3:q_proj": torch.ones(4)}},
                "slices": {"3:in_proj_qkv": {"q": (0, 2), "k": (2, 4), "v": (4, 10)}}}
        v = get_band_offsets(data, "in_proj_qkv.v", "all")["3:in_proj_qkv"]
        self.assertEqual(v[:4].tolist(), [0, 0, 0, 0])
        self.assertEqual(v[4:].tolist(), list(range(5, 11)))
        qk = get_band_offsets(data, "in_proj_qkv.qk", "all")["3:in_proj_qkv"]
        self.assertEqual(qk[4:].abs().sum().item(), 0)
        whole = get_band_offsets(data, "in_proj_qkv", "all")["3:in_proj_qkv"]
        self.assertEqual(whole.tolist(), list(range(1, 11)))
        rand = random_matched_offsets({"3:in_proj_qkv": v}, 0, band_supports(data, "in_proj_qkv.v"))["3:in_proj_qkv"]
        self.assertEqual(rand[:4].abs().sum().item(), 0)
        self.assertAlmostEqual(rand.norm().item(), v.norm().item(), places=4)
        self.assertIsNone(band_supports(data, "in_proj_qkv"))

    def test_unsliced_randoms_are_unchanged_by_the_supports_argument(self):
        offsets = {"1:v_proj": torch.full((16,), 0.3), "2:v_proj": torch.full((16,), -0.2)}
        a, b = random_matched_offsets(offsets, 4), random_matched_offsets(offsets, 4, None)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))


class HybridBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_deltanet_projection_is_found_and_steered(self):
        model = tiny_hybrid()
        batch = {"input_ids": torch.tensor([[1, 4, 5, 9]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        layer = get_decoder_layers(model)[0]
        self.assertIs(attn_proj(layer, "in_proj_qkv"), layer.linear_attn.in_proj_qkv)
        width = layer.linear_attn.in_proj_qkv.out_features
        base, _ = capture_proj(model, 0, "in_proj_qkv", batch)
        with apply_module_steering(model, {"0:in_proj_qkv": torch.full((width,), 0.25)}, 2.0):
            steered, _ = capture_proj(model, 0, "in_proj_qkv", batch)
        self.assertTrue(torch.allclose(steered - base, torch.full_like(base, 0.5), atol=1e-5))

    def test_positional_modes_survive_hybrid_generation(self):
        model = tiny_hybrid()
        width = get_decoder_layers(model)[0].linear_attn.in_proj_qkv.out_features
        for mode in ("content", "from_last"):
            positions = PositionalSteering(None, mode)
            seen = []
            with apply_module_steering(model, {"0:in_proj_qkv": torch.full((width,), 0.25)}, 1.0, positions):
                h = attn_proj(get_decoder_layers(model)[0], "in_proj_qkv").register_forward_hook(
                    lambda _m, _i, o: seen.append(positions.mask(o)[..., 0].clone()))
                if mode == "content":
                    positions.set_spans([(1, 3)])
                with torch.no_grad():
                    model.generate(input_ids=torch.tensor([[1, 4, 5, 9]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                                   max_new_tokens=3, do_sample=False, pad_token_id=0)
                h.remove()
            if mode == "content":
                self.assertEqual(seen[0].tolist(), [[0.0, 1.0, 1.0, 0.0]])
                self.assertTrue(all(m.sum() == 0 for m in seen[1:]))
            else:
                self.assertEqual(seen[0].tolist(), [[0.0, 0.0, 0.0, 1.0]])
                self.assertTrue(all(m.tolist() == [[1.0]] for m in seen[1:]))

    def test_extraction_records_the_fused_slices(self):
        mod = load_script()
        peft = hybrid_lora(tiny_hybrid())
        modules = mod.band_lora_modules(peft, {0, 1, 3}, ("in_proj_qkv", "v_proj"))
        self.assertEqual(sorted(modules), ["0:in_proj_qkv", "1:in_proj_qkv", "3:v_proj"])
        slices = mod.fused_slices(peft, modules)
        self.assertEqual(sorted(slices), ["0:in_proj_qkv", "1:in_proj_qkv"])
        self.assertEqual(slices["0:in_proj_qkv"], {"q": (0, 16), "k": (16, 32), "v": (32, 64)})


class ReasoningAndBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_answer_text(self):
        self.assertEqual(answer_text("<t>the garage?</t> kitchen", "<t>", "</t>"), " kitchen")
        self.assertIsNone(answer_text("<t>the garage? maybe", "<t>", "</t>"))
        self.assertEqual(answer_text("kitchen", "<t>", "</t>"), "kitchen")
        self.assertEqual(answer_text("<t>a</t> b</t> c", "<t>", "</t>"), " c")

    def test_cli_content_mode_batching_and_think_markers(self):
        from selfconcept.soo import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            tokenizer = save_tiny_model_with_tokenizer(tiny_model(), model_path)
            text = tokenizer.apply_chat_template([{"role": "user", "content": "You Bob room"}], add_generation_prompt=True, tokenize=False)
            ids, span = content_span(tokenizer, text, "You Bob room")
            self.assertEqual(span, (0, 3))
            self.assertEqual(ids, [10, 11, 12, 9])
            offsets = {"1:v_proj": torch.full((16,), 0.3)}
            deltas = root / "band_deltas.pt"
            torch.save({"model": str(model_path), "offsets": {"last": offsets, "all": offsets}, "stats": {}}, deltas)
            data = root / "data"
            data.mkdir()
            prompts = ["You Bob room", "You room", "Bob Bob room kitchen"]
            rows = [{"scenario": "main", "example_id": f"main_{i}", "prompt": p, "other_name": "Bob",
                     "honest_answer": "kitchen", "deceptive_answer": "garage"} for i, p in enumerate(prompts)]
            (data / "main.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = root / "eval"
            common = ["--model", str(model_path), "--data", str(data), "--out", str(out), "--scenarios", "main",
                      "--suffix", "none", "--max-new-tokens", "3", "--band-deltas", str(deltas), "--band-token-mode", "all"]
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), mock.patch.object(evaluate, "pick_device", lambda: "cpu"):
                for tag, extra in [("one", []), ("batched", ["--batch-size", "2"]),
                                   ("think", ["--batch-size", "3", "--think-open", "self", "--think-close", "other"])]:
                    with mock.patch("sys.argv", ["evaluate", *common, "--tag", tag, "--steer-positions", "content", *extra]):
                        evaluate.main()
            one = [json.loads(l) for l in (out / "one_main_none.jsonl").read_text().splitlines()]
            batched = [json.loads(l) for l in (out / "batched_main_none.jsonl").read_text().splitlines()]
            self.assertEqual([r["response"] for r in one], [r["response"] for r in batched])
            summary = json.loads((out / "think_main_none_summary.json").read_text())
            self.assertIn("truncated", summary["counts"])
            self.assertEqual(summary["batch_size"], 3)
            self.assertEqual(summary["steering"]["positions"], "content")


if __name__ == "__main__":
    unittest.main()


class KimiSweepRuleTests(unittest.TestCase):
    """KIMI_SWEEP_ROUND.md: an adapter works only if it lowers deception and raises honesty in both orientations."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import sys
        root = Path(__file__).resolve().parents[1]
        script = next((c for c in (root / "experiments/soo/scripts/analyze_kimi_sweep.py", root / "analyze_kimi_sweep.py") if c.exists()), None)
        if script is None:
            raise unittest.SkipTest("analyze_kimi_sweep.py is not part of this snapshot")
        sys.path.insert(0, str(script.parent))
        spec = importlib.util.spec_from_file_location("analyze_kimi_sweep", script)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    @staticmethod
    def runs(adapter):
        cell = lambda h, d, t=0.0: {"main": {"rates": {"honest": h, "deceptive": d, "refusal": 0.0, "other": 0.0, "truncated": t}}}  # noqa: E731
        return {"base": {"orig": cell(0.32, 0.68), "mirrored": cell(0.24, 0.75)},
                "L8-adapter": {o: cell(*v) for o, v in adapter.items()}}

    def label(self, **adapter):
        return self.mod.screen(self.runs(adapter), "L8-adapter")["label"]

    def test_labels(self):
        self.assertEqual(self.label(orig=(0.9, 0.1), mirrored=(0.8, 0.2)), "working")
        self.assertEqual(self.label(orig=(0.0, 1.0), mirrored=(0.03, 0.97)), "worse")
        self.assertEqual(self.label(orig=(0.3, 0.7), mirrored=(0.2, 0.8)), "null")
        self.assertEqual(self.label(orig=(0.9, 0.1), mirrored=(0.24, 0.75)), "mixed")
        self.assertEqual(self.label(orig=(0.9, 0.1)), "pending")

    def test_a_drop_into_truncation_is_not_working(self):
        self.assertEqual(self.label(orig=(0.35, 0.2, 0.45), mirrored=(0.3, 0.2, 0.5)), "mixed (damage)")
        self.assertFalse(self.mod.screen(self.runs({"orig": (0.6, 0.1, 0.3), "mirrored": (0.6, 0.1, 0.3)}), "L8-adapter")["working"])

    def test_confirmation_needs_perspectives_in_both_orientations(self):
        runs = self.runs({"orig": (0.9, 0.1), "mirrored": (0.8, 0.2)})
        self.assertIsNone(self.mod.confirm(runs, "L8-adapter"))
        for o, p in (("orig", 1.0), ("mirrored", 0.85)):
            runs["L8-adapter"][o]["perspectives"] = {"rates": {"honest": p}}
        self.assertFalse(self.mod.confirm(runs, "L8-adapter"))
