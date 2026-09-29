"""Software checks for band steering: a LoRA band's mean deltas as fixed offsets (STEER_ROUND.md).

Run: .venv/bin/python -m unittest discover -s tests -p 'test_soo_band_steer.py' -v
Tiny random Llama with a random LoRA on CPU; no model download. These check the
delta measurement and the multi-module hook, not any experimental result.
"""

import importlib.util
from pathlib import Path
import unittest

import torch

from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.steering import (
    PositionalSteering, apply_module_steering, attn_proj, get_band_offsets, offset_key, random_matched_offsets,
    steer_modules,
)
from test_soo_lora_subset import lora_model, tiny_model


def load_script():
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "experiments/soo/scripts/extract_band_deltas.py", root / "extract_band_deltas.py"]  # repo, frozen snapshot
    script = next(c for c in candidates if c.exists())
    import sys
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))  # the script imports make_constants from its own directory
    spec = importlib.util.spec_from_file_location("extract_band_deltas", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture_proj(model, layer, name, batch):
    store = []
    handle = attn_proj(get_decoder_layers(model)[layer], name).register_forward_hook(lambda _m, _i, o: store.append(o.detach().clone()))
    try:
        with torch.no_grad():
            out = model(**batch)
    finally:
        handle.remove()
    return store[-1], out.logits


class BandDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.peft = lora_model()
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 0], [1, 6, 9, 0, 0]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])}
        self.mod = load_script()

    def test_measured_deltas_equal_the_lora_path(self):
        modules = self.mod.band_lora_modules(self.peft, {1, 2}, ("v_proj",))
        self.assertEqual(sorted(modules), ["1:v_proj", "2:v_proj"])
        # Reference: s * B(A(x)) from the LoRA weights, captured on the module input.
        inputs = {}
        handles = [m.register_forward_hook(lambda _m, i, _o, k=k: inputs.__setitem__(k, i[0].detach().clone())) for k, m in modules.items()]
        with torch.no_grad():
            self.peft(**self.batch)
        for h in handles:
            h.remove()
        offsets, stats = self.mod.measure_band(self.peft, modules, [self.batch])
        mask = self.batch["attention_mask"].bool()
        for key, module in modules.items():
            x = inputs[key]
            with torch.no_grad():
                delta = module.lora_B["default"](module.lora_A["default"](x)) * module.scaling["default"]
            expected_last = torch.stack([delta[0, 3], delta[1, 2]]).mean(0)
            expected_all = delta[mask].mean(0)
            torch.testing.assert_close(offsets["last"][key], expected_last, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(offsets["all"][key], expected_all, atol=1e-5, rtol=1e-4)
            self.assertEqual(stats["last"][key]["n_prompts"], 2)
            self.assertEqual(stats["all"][key]["n_tokens"], 7)
            self.assertGreater(stats["all"][key]["rms_spread"], 0.0)
            self.assertGreater(stats["last"][key]["base_output_norm"], 0.0)
        self.assertFalse(any(m._forward_hooks for m in self.peft.modules()))


class SteerModulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = tiny_model()
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 0], [1, 6, 9, 0, 0]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])}
        torch.manual_seed(3)
        self.offsets = {offset_key(1, "v_proj"): torch.randn(16), offset_key(2, "v_proj"): torch.randn(16)}

    def test_offsets_added_at_every_position_in_every_band_layer(self):
        base1, logits0 = capture_proj(self.model, 1, "v_proj", self.batch)
        base2, _ = capture_proj(self.model, 2, "v_proj", self.batch)
        with apply_module_steering(self.model, self.offsets, 2.0):
            out1, logits1 = capture_proj(self.model, 1, "v_proj", self.batch)
            out2, _ = capture_proj(self.model, 2, "v_proj", self.batch)
        torch.testing.assert_close(out1 - base1, (2.0 * self.offsets["1:v_proj"]).expand_as(base1))
        # Layer 2's input already changed, so compare its delta on the hook output only in shape terms.
        self.assertEqual(out2.shape, base2.shape)
        self.assertFalse(torch.allclose(logits0, logits1))
        # Cleanup restores the base model exactly.
        after, logits2 = capture_proj(self.model, 1, "v_proj", self.batch)
        torch.testing.assert_close(after, base1)
        torch.testing.assert_close(logits2, logits0)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))

    def test_response_positions_only(self):
        base1, _ = capture_proj(self.model, 1, "v_proj", self.batch)
        positions = PositionalSteering([9], "response")  # token 9 marks the response start
        with apply_module_steering(self.model, {"1:v_proj": self.offsets["1:v_proj"]}, 1.0, positions):
            out1, _ = capture_proj(self.model, 1, "v_proj", self.batch)
        diff = out1 - base1
        v = self.offsets["1:v_proj"]
        torch.testing.assert_close(diff[0, :3], torch.zeros(3, 16))
        torch.testing.assert_close(diff[0, 3:], v.expand(2, 16))
        torch.testing.assert_close(diff[1, :2], torch.zeros(2, 16))
        torch.testing.assert_close(diff[1, 2:], v.expand(3, 16))

    def test_width_check_and_random_controls(self):
        with self.assertRaises(ValueError):
            steer_modules(self.model, {"1:v_proj": torch.ones(8)}, 1.0)
        r0, r1 = random_matched_offsets(self.offsets, 0), random_matched_offsets(self.offsets, 1)
        for key, v in self.offsets.items():
            self.assertAlmostEqual(r0[key].norm().item(), v.norm().item(), places=5)
            self.assertFalse(torch.allclose(r0[key], r1[key]))
            self.assertFalse(torch.allclose(r0[key], v))
        self.assertFalse(torch.allclose(r0["1:v_proj"], r0["2:v_proj"]))
        torch.testing.assert_close(random_matched_offsets(self.offsets, 0)["1:v_proj"], r0["1:v_proj"])

    def test_get_band_offsets_filters_by_module(self):
        data = {"offsets": {"last": {"1:v_proj": torch.ones(16), "1:q_proj": torch.zeros(16)}, "all": {}}}
        self.assertEqual(list(get_band_offsets(data, "v_proj", "last")), ["1:v_proj"])
        with self.assertRaises(KeyError):
            get_band_offsets(data, "k_proj", "last")
        with self.assertRaises(ValueError):
            get_band_offsets(data, "v_proj", "mean")


def save_tiny_model_with_tokenizer(model, path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    model.save_pretrained(path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "self": 4, "other": 5, "longer": 6, "0": 7, "1": 8, "assistant": 9,
             "You": 10, "Bob": 11, "room": 12, "kitchen": 13, "garage": 14}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
    tokenizer.save_pretrained(path)
    return tokenizer


class CliTests(unittest.TestCase):
    """The two entry points accept band deltas: evaluate steers the base model, measure_overlap measures it."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_evaluate_and_measure_overlap_with_band_deltas(self):
        import json
        import os
        import tempfile
        from contextlib import chdir
        from unittest import mock

        from selfconcept.soo import evaluate, measure_overlap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            save_tiny_model_with_tokenizer(tiny_model(), model_path)  # same seed as lora_model()'s base
            peft = lora_model()
            peft.save_pretrained(root / "adapter")
            offsets = {"1:v_proj": torch.full((16,), 0.3), "2:v_proj": torch.full((16,), -0.2), "1:q_proj": torch.ones(16)}
            deltas = root / "band_deltas.pt"
            torch.save({"model": str(model_path), "offsets": {"last": offsets, "all": offsets}, "stats": {}}, deltas)
            data = root / "data"
            data.mkdir()
            rows = [{"scenario": "main", "example_id": f"main_{i}", "prompt": "You Bob room", "other_name": "Bob",
                     "honest_answer": "kitchen", "deceptive_answer": "garage"} for i in range(2)]
            (data / "main.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = root / "eval"
            common = ["--model", str(model_path), "--data", str(data), "--out", str(out), "--scenarios", "main", "--n", "2",
                      "--suffix", "none", "--max-new-tokens", "3"]
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), mock.patch.object(evaluate, "pick_device", lambda: "cpu"):
                with mock.patch("sys.argv", ["evaluate", *common, "--tag", "base"]):
                    evaluate.main()
                with mock.patch("sys.argv", ["evaluate", *common, "--tag", "band", "--band-deltas", str(deltas),
                                             "--band-alpha", "2", "--steer-positions", "response"]):
                    evaluate.main()
                with mock.patch("sys.argv", ["evaluate", *common, "--tag", "rand", "--band-deltas", str(deltas),
                                             "--band-module", "q_proj", "--band-random-seed", "1"]):
                    evaluate.main()
            band = json.loads((out / "band_main_none_summary.json").read_text())["steering"]
            self.assertEqual(band["layers"], [1, 2])
            self.assertEqual(band["alpha"], 2.0)
            self.assertEqual(band["positions"], "response")
            self.assertAlmostEqual(band["offset_norms"]["1:v_proj"], offsets["1:v_proj"].norm().item(), places=4)
            rand = json.loads((out / "rand_main_none_summary.json").read_text())["steering"]
            self.assertEqual(rand["random_seed"], 1)
            self.assertEqual(list(rand["offset_norms"]), ["1:q_proj"])
            self.assertIsNone(json.loads((out / "base_main_none_summary.json").read_text())["steering"])

            probes = root / "probes.jsonl"
            pairs = [{"id": str(i), "family": f"f{i}", "split": "development", "kind": kind,
                      "self_prompt": "self 0 longer", "other_prompt": f"other {i} longer"}
                     for i, kind in enumerate(("self_other", "nonsocial"))]
            probes.write_text("".join(json.dumps(r) + "\n" for r in pairs))
            run = root / "overlap"
            measure_overlap.main(["--model", str(model_path), "--probes", str(probes), "--out", str(run), "--layer", "2",
                                  "--band-deltas", str(deltas), "--band-modules", "v_proj", "q_proj", "--band-alpha", "1",
                                  "--band-random-seeds", "0", "--positions", "response", "--adapter", str(root / "adapter"),
                                  "--adapter-layers", "range:1-2", "--adapter-modules", "v_proj", "--bootstrap", "50"])
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["conditions"], ["base", "band_v_proj", "band_v_proj_random_s0", "band_q_proj", "adapter"])
            self.assertEqual(manifest["condition_settings"]["adapter"]["subset"]["kept"], 2)
            self.assertEqual(manifest["condition_settings"]["band_v_proj"]["layers"], [1, 2])
            acts = torch.load(run / "activations.pt", weights_only=True)
            self.assertFalse(torch.allclose(acts["base"]["hook_before"], acts["band_v_proj"]["hook_before"]))
            self.assertFalse(torch.allclose(acts["band_v_proj"]["hook_before"], acts["band_v_proj_random_s0"]["hook_before"]))


if __name__ == "__main__":
    unittest.main()
