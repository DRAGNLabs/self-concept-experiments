"""Separate measurement modes, signed projections, and reward-hacking joins."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from selfconcept.assistant_axis.projection import load_unit_axes, unit_axis_scorer
from selfconcept.correlation.analyze import analyze_run, CompletedAnalysis
from selfconcept.correlation.generate import GenerationSettings, measured_generate, MeasuredModel
from selfconcept.correlation.run import main, parse_args
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer
from selfconcept.common.jsonl import read_jsonl
from selfconcept.measurement.capture import capture
from selfconcept.measurement.interface import Measurement


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
            with capture(model, [1]) as hidden, capture(model, [1], unit_axis_scorer({1: {"direction": direction}})) as projected:
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
            measurement = Measurement("assistant-axis", [1], 1, unit_axis_scorer({1: {"direction": direction}}))
            rec = measured_generate(MeasuredModel(model, tok, "qwen", measurement),
                                    GenerationSettings(out, {"do_sample": False}, 1729, 4),
                                    "impossible_conflicting", "example", [{"role": "user", "content": "A B"}], 0)
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
                  patch("selfconcept.cotness.probe.train") as train,
                  patch("selfconcept.cotness.probe.load") as load_probe,
                  patch("selfconcept.correlation.run.run_scenarios") as run_scenarios):
                main()
                train.assert_not_called()
                load_probe.assert_not_called()
                self.assertEqual(run_scenarios.call_args.args[1].measurement.name, "assistant-axis")
                self.assertFalse((out / "probes").exists())
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(manifest["stage"], "complete")
                self.assertEqual(manifest["assistant_axis"]["primary_layer"], 0)
                self.assertEqual(manifest["measurement"], {"name": "assistant-axis", "layers": [0], "primary_layer": 0})
                self.assertTrue((out / "assistant_axis.pt").exists())
                main()  # Identical run is resumable.
                torch.save(-torch.ones(3, 16), axis)
                with self.assertRaisesRegex(ValueError, "different configurations"):
                    main()
                self.assertEqual(load_model.call_count, 2)

    def test_transcript_run_scores_resumes_and_analyzes(self):
        model = self.model()
        tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2,
                                        "user:": 3, "A": 4, "B": 5, "assistant:": 6, "<unk>": 7},
                                       unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                      eos_token="<eos>", unk_token="<unk>")
        tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}assistant:"
        messages = [{"role": "user", "content": "A B"}]
        direction = np.ones(16, dtype=np.float32) / 4
        measurement = Measurement("assistant-axis", [0], 0, unit_axis_scorer({0: {"direction": direction}}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_out = root / "source"
            source_out.mkdir()
            source = measured_generate(
                MeasuredModel(model, cast(HFTokenizer, tok), "qwen", measurement),
                GenerationSettings(source_out, {"do_sample": False}, 1729, 4),
                "impossible_conflicting", "example", cast(Conversation, messages), 0)
            transcripts = root / "transcripts.jsonl"
            transcripts.write_text(json.dumps({
                "example_id": "example", "scenario": "impossible_conflicting", "turn": 0,
                "messages": messages, "chat_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
                "raw_response": source["raw_response"], "truncated": source["truncated"],
                "outcome": {"example_id": "example", "scenario": "impossible_conflicting",
                            "status": "complete", "label": "cheat_modify_tests", "model_key": "test/model"}
            }) + "\n")
            transcript_args = ["--model", "gemma4-12b", "--out", str(root / "unused"),
                               "--transcripts", str(transcripts)]
            for excluded in (["--train-only"], ["--scenarios", "main"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(transcript_args + excluded)
            axis = root / "axis.pt"
            torch.save(torch.ones(3, 16), axis)
            out = root / "run"
            args = parse_args(["--model", "test/model", "--family", "qwen", "--out", str(out),
                               "--measurement", "assistant-axis", "--assistant-axis", str(axis),
                               "--transcripts", str(transcripts)])
            self.assertEqual(args.scenarios, ["impossible_conflicting"])
            with (patch("selfconcept.correlation.run.parse_args", return_value=args),
                  patch("selfconcept.correlation.run.AutoTokenizer.from_pretrained", return_value=tok),
                  patch("selfconcept.correlation.run.load_causal_lm", return_value=model)):
                main()
                main()
                transcripts.write_text(transcripts.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "different configurations"):
                    main()
            scored_generations = read_jsonl(out / "generations.jsonl")
            self.assertEqual(len(scored_generations), 1)
            scored = scored_generations[0]
            for layer, regions in source["scores"].items():
                for region, summary in regions.items():
                    self.assertEqual(scored["scores"][layer][region]["n"], summary["n"])
                    if summary["mean"] is not None:
                        self.assertAlmostEqual(scored["scores"][layer][region]["mean"], summary["mean"], places=5)
            self.assertEqual(len(read_jsonl(out / "outcomes.jsonl")), 1)
            self.assertEqual(len(read_jsonl(out / "base_impossible_conflicting.jsonl")), 1)
            result = cast(CompletedAnalysis, analyze_run(out, bootstrap_samples=10))
            self.assertEqual(result["status"], "exploratory")
            self.assertEqual(result["coverage"]["impossible_conflicting"]["scored_positive"], 1)

    def test_cotness_run_records_measurement_block_only_for_usable_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.jsonl"
            corpus.write_text("")
            for usable in (True, False):
                out = root / f"usable-{usable}"
                (out / "probes").mkdir(parents=True)
                (out / "probes/validation.json").write_text(json.dumps(
                    {"usable": usable, "primary_layer": 1, "layers": {"0": {}, "1": {}}}))
                for layer in (0, 1):
                    np.savez(out / f"probes/layer-{layer}.npz", weight=np.zeros((3, 16), dtype=np.float32),
                             bias=np.zeros(3, dtype=np.float32))
                args = parse_args(["--model", "gemma4-12b", "--out", str(out), "--corpus", str(corpus)])
                with (patch("selfconcept.correlation.run.parse_args", return_value=args),
                      patch("selfconcept.correlation.run.AutoTokenizer.from_pretrained"),
                      patch("selfconcept.correlation.run.load_causal_lm", return_value=self.model()),
                      patch("selfconcept.cotness.probe.train") as train,
                      patch("selfconcept.correlation.run.run_scenarios") as run_scenarios):
                    main()
                train.assert_not_called()
                manifest = json.loads((out / "manifest.json").read_text())
                if usable:
                    measurement = run_scenarios.call_args.args[1].measurement
                    self.assertEqual((measurement.name, measurement.layers, measurement.primary_layer), ("cotness", [0, 1], 1))
                    torch.testing.assert_close(measurement.score(layer=1, residual=torch.ones(2, 16)), torch.full((2,), 1/3))
                    self.assertEqual(manifest["measurement"], {"name": "cotness", "layers": [0, 1], "primary_layer": 1})
                else:
                    run_scenarios.assert_not_called()
                    self.assertNotIn("measurement", manifest)
                    (out / "outcomes.jsonl").write_text("")
                    self.assertEqual(analyze_run(out)["status"], "probe_failed_validation")

    def test_analysis_reads_primary_layer_from_measurement_block(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "manifest.json").write_text(json.dumps({"args": {"measurement": "cotness"},
                                                         "measurement": {"name": "cotness", "layers": [1, 2], "primary_layer": 2}}))
            (out / "outcomes.jsonl").write_text("")
            result = analyze_run(out, bootstrap_samples=10)
            self.assertEqual(result["primary_layer"], 2)
            self.assertFalse((out / "probes").exists())

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
                                            "scores": {"1": {"final": {"mean": i-4 if turn == 0 else 0, "n": 10}}}})
                (out / f"base_{scenario}_graded.jsonl").write_text("".join(json.dumps(g)+"\n" for g in grades))
                grades = []
            outcomes += [{"scenario": "impossible_oneoff", "example_id": "missing", "status": "truncated",
                          "label": "cheat_modify_tests"}, outcomes[0]]
            generations.append(generations[0])
            (out / "outcomes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in outcomes))
            (out / "generations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in generations))
            result = analyze_run(out, bootstrap_samples=10)
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
                analyze_run(out, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
