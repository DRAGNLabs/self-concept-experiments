"""Separate measurement modes, signed projections, and reward-hacking joins."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from selfconcept.assistant_axis.projection import load_unit_axes
from selfconcept.correlation.analyze import analyze
from selfconcept.cotness.probe import capture
from selfconcept.correlation.run import MeasuredGenerator, main, parse_args


class AxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(8)
        return LlamaForCausalLM(LlamaConfig(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=2, bos_token_id=1,
            eos_token_id=None, pad_token_id=0)).eval()

    def test_load_existing_tensor_and_wrapped_axis_preserves_sign(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "axis.pt"
            axis = torch.zeros(3, 16)
            axis[:, 0] = -4
            for data in (axis, {"axis": axis, "model": "test/model"}):
                torch.save(data, path)
                fitted = load_unit_axes(path, [1, 2], 3, 16, "test/model")
                np.testing.assert_array_equal(fitted[1]["direction"], [-1] + [0]*15)
            with self.assertRaisesRegex(ValueError, "different model"):
                load_unit_axes(path, [1], 3, 16, "other/model")

    def test_invalid_axis_dimensions_layers_and_directions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "axis.pt"
            for axis, layers in [(torch.ones(16), [1]), (torch.ones(2, 16), [1]),
                                 (torch.ones(3, 15), [1]), (torch.ones(3, 16), [-1]),
                                 (torch.ones(3, 16), [3]), (torch.ones(3, 16), [1, 1]),
                                 (torch.zeros(3, 16), [1]),
                                 (torch.full((3, 16), float("nan")), [1]),
                                 (torch.full((3, 16), float("inf")), [1])]:
                with self.subTest(shape=axis.shape, layers=layers):
                    torch.save(axis, path)
                    with self.assertRaises(ValueError):
                        load_unit_axes(path, layers, 3, 16)

    def test_axis_hook_equals_signed_dot_product_and_does_not_change_generation(self):
        model = self.model()
        inputs = torch.tensor([[1, 3, 4]])
        direction = np.zeros(16, dtype=np.float32)
        direction[0] = -1
        with torch.inference_mode():
            baseline = model.generate(inputs, max_new_tokens=4, do_sample=False)
            with capture(model, [1]) as hidden, capture(model, [1], {1: {"direction": direction}}) as projected:
                measured = model.generate(inputs, max_new_tokens=4, do_sample=False)
        self.assertTrue(torch.equal(baseline, measured))
        values = torch.cat(projected[1])
        torch.testing.assert_close(values, -torch.cat(hidden[1])[:, 0])
        self.assertEqual(len(values), measured.shape[1]-1)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_generator_writes_only_selected_metric_and_exact_token_traces(self):
        model = self.model()
        tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2,
                                        "user:": 3, "A": 4, "B": 5, "assistant:": 6, "<unk>": 7},
                                       unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                     eos_token="<eos>", unk_token="<unk>")
        tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}assistant:"
        direction = np.zeros(16, dtype=np.float32)
        direction[0] = -1
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            generator = MeasuredGenerator(model, tok, "qwen", {1: {"direction": direction}},
                                          out, 4, measurement="assistant-axis")
            generator.scenario = "impossible_conflicting"
            generator([{"role": "user", "content": "A B"}], 0, "example")
            rec = generator.records[0]
            self.assertEqual(rec["measurement"], "assistant-axis")
            self.assertEqual(rec["generated_alignment"], "exact")
            with np.load(out / rec["trace"]) as trace:
                self.assertIn("assistant_axis_layer_1", trace.files)
                self.assertNotIn("cotness_layer_1", trace.files)
                self.assertEqual(len(trace["assistant_axis_layer_1"]), len(trace["input_ids"])-1)
                with torch.inference_mode(), capture(model, [1]) as hidden:
                    model(torch.tensor([trace["input_ids"][:-1].tolist()]))
                np.testing.assert_allclose(trace["assistant_axis_layer_1"],
                                           -torch.cat(hidden[1])[:, 0].numpy(), atol=1e-6)
                self.assertEqual(rec["scores"]["1"]["prompt"]["n"], 2)
                self.assertAlmostEqual(rec["scores"]["1"]["prompt"]["mean"],
                                       float(trace["assistant_axis_layer_1"][trace["indices_prompt"]].mean()))

    def test_cli_modes_are_exclusive_and_cotness_remains_default(self):
        base = ["--model", "gemma4-12b", "--out", "unused"]
        self.assertEqual(parse_args(base).measurement, "cotness")
        axis_args = base + ["--measurement", "assistant-axis", "--assistant-axis", "axis.pt"]
        args = parse_args(axis_args)
        self.assertEqual(set(args.scenarios), {"impossible_original", "impossible_oneoff", "impossible_conflicting"})
        for extra in (["--assistant-axis", "axis.pt"], ["--measurement", "assistant-axis"],
                      ["--axis-layers", "1"],
                      axis_args[4:] + ["--probe-source", "probes"],
                      axis_args[4:] + ["--train-only"],
                      axis_args[4:] + ["--axis-layers", "-1"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + extra)
        custom = parse_args(["--model", "allenai/Olmo-3.1-32B-Think", "--family", "olmo",
                             "--out", "unused", "--measurement", "assistant-axis", "--assistant-axis", "axis.pt"])
        self.assertEqual(custom.family, "olmo")

    def test_generator_rejects_mixed_or_mislabeled_projections(self):
        axis = {"direction": np.ones(16, dtype=np.float32)}
        cot = {"weight": np.zeros((3, 16), dtype=np.float32), "bias": np.zeros(3, dtype=np.float32)}
        for measurement, projections in [("cotness", {1: axis}), ("assistant-axis", {1: cot}),
                                          ("assistant-axis", {1: axis, 2: cot})]:
            with self.subTest(measurement=measurement), self.assertRaisesRegex(ValueError, "only the selected"):
                MeasuredGenerator(None, None, "qwen", projections, Path("unused"), 4, measurement=measurement)

    def test_axis_run_never_trains_or_loads_cotness_and_rejects_changed_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            axis = root / "axis.pt"
            torch.save(torch.ones(3, 16), axis)
            out = root / "run"
            args = parse_args(["--model", "test/model", "--family", "qwen", "--out", str(out),
                               "--measurement", "assistant-axis", "--assistant-axis", str(axis),
                               "--corpus", str(root / "nonexistent.jsonl")])
            with (patch("selfconcept.correlation.run.parse_args", return_value=args),
                  patch("selfconcept.correlation.run.AutoTokenizer.from_pretrained"),
                  patch("selfconcept.correlation.run.load_causal_lm", return_value=self.model()) as load_model,
                  patch("selfconcept.correlation.run.probe.train") as train,
                  patch("selfconcept.correlation.run.probe.load") as load_probe,
                  patch("selfconcept.correlation.run.evaluate") as evaluate):
                main()
                train.assert_not_called()
                load_probe.assert_not_called()
                self.assertIn("direction", evaluate.call_args.args[3][0])
                self.assertFalse((out / "probes").exists())
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(manifest["stage"], "complete")
                self.assertEqual(manifest["assistant_axis"]["primary_layer"], 0)
                self.assertTrue((out / "assistant_axis.pt").exists())
                main()  # Identical run is resumable.
                torch.save(-torch.ones(3, 16), axis)
                with self.assertRaisesRegex(ValueError, "different configurations"):
                    main()
                self.assertEqual(load_model.call_count, 2)

    def test_axis_analysis_uses_first_turn_reward_hacking_and_no_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "manifest.json").write_text(json.dumps({"args": {"measurement": "assistant-axis"},
                                                         "assistant_axis": {"primary_layer": 1}}))
            outcomes, generations, grades = [], [], []
            for scenario in ("impossible_oneoff", "impossible_conflicting", "impossible_original"):
                for i in range(8):
                    outcomes.append({"scenario": scenario, "example_id": str(i), "status": "complete",
                                     "label": "cheat_modify_tests" if i < 4 else "flagged"})
                    grades.append({"example_id": str(i), "label": "no_code"})
                    for turn in (0, 1):
                        generations.append({"scenario": scenario, "example_id": str(i), "turn": turn,
                                            "measurement": "assistant-axis", "prompt_tokens": 10,
                                            "scores": {"1": {"prompt": {"mean": i-4 if turn == 0 else 0, "n": 10}}}})
                (out / f"base_{scenario}_graded.jsonl").write_text("".join(json.dumps(g)+"\n" for g in grades))
                grades = []
            outcomes += [{"scenario": "impossible_oneoff", "example_id": "missing", "status": "truncated",
                          "label": "cheat_modify_tests"}, outcomes[0]]
            generations.append(generations[0])
            (out / "outcomes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in outcomes))
            (out / "generations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in generations))
            result = analyze(out, bootstrap=10)
            self.assertEqual(result["measurement"], "assistant-axis")
            self.assertEqual(len(result["cells"]), 2)
            for cell in result["cells"]:
                self.assertTrue(cell["primary"])
                self.assertEqual(cell["n"], 8)
                self.assertEqual(cell["auc"], 0.)
                self.assertLess(cell["rho"], 0.)
            self.assertEqual(result["coverage"]["impossible_original"]["capability_control"], 8)
            self.assertEqual(result["coverage"]["impossible_oneoff"]["unscored"], 1)
            generations[0]["measurement"] = "cotness"
            (out / "generations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in generations))
            with self.assertRaisesRegex(ValueError, "mixed measurements"):
                analyze(out, bootstrap=10)


if __name__ == "__main__":
    unittest.main()
