"""Software checks for constant replacement and the from_last position mask.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_soo_replace.py' -v
Controlled tensors and a tiny random Llama on CPU; no model download. These
check the hook, not any experimental result.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from selfconcept.soo.activations import attn_out_proj, get_decoder_layers
from selfconcept.soo.steering import (
    PositionalSteering, apply_steering, get_constant, load_constants, random_matched_vector, steer_o_proj,
)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=2, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return LlamaForCausalLM(config).cpu().eval()


def capture(model, layer, batch, **kw):
    store = []
    handle = attn_out_proj(get_decoder_layers(model)[layer]).register_forward_hook(lambda _m, _i, o: store.append(o.detach().clone()))
    try:
        with torch.no_grad():
            out = model(**batch, **kw)
    finally:
        handle.remove()
    return store[-1], out


class ReplaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = tiny_model()
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 0], [1, 6, 9, 0, 0]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])}
        self.c = torch.linspace(-0.3, 0.4, 16)
        self.base, _ = capture(self.model, 1, self.batch)

    def test_replace_all_positions_sets_every_output_to_the_constant(self):
        with apply_steering(self.model, 1, self.c, 1.0, "replace"):
            after, _ = capture(self.model, 1, self.batch)
        torch.testing.assert_close(after, self.c.expand_as(after))

    def test_replace_from_last_touches_only_the_last_valid_token(self):
        with apply_steering(self.model, 1, self.c, 1.0, "replace", PositionalSteering(None, "from_last")):
            after, _ = capture(self.model, 1, self.batch)
        for row, last in ((0, 3), (1, 2)):
            torch.testing.assert_close(after[row, last], self.c)
            torch.testing.assert_close(after[row, :last], self.base[row, :last], rtol=0, atol=0)

    def test_partial_alpha_interpolates(self):
        with apply_steering(self.model, 1, self.c, 0.25, "replace"):
            after, _ = capture(self.model, 1, self.batch)
        torch.testing.assert_close(after, 0.75 * self.base + 0.25 * self.c)

    def test_from_last_covers_generated_tokens_through_the_cache(self):
        prompt = {"input_ids": torch.tensor([[1, 4, 5, 9]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        pos = PositionalSteering(None, "from_last")
        handle = steer_o_proj(self.model, 1, self.c, 1.0, "replace", pos)
        try:
            with torch.no_grad():
                first, out = capture(self.model, 1, prompt, use_cache=True)
                torch.testing.assert_close(first[0, 3], self.c)
                step = {"input_ids": torch.tensor([[7]]), "attention_mask": torch.ones(1, 5, dtype=torch.long),
                        "past_key_values": out.past_key_values}
                second, _ = capture(self.model, 1, step, use_cache=True)
            torch.testing.assert_close(second[0, 0], self.c)
            self.assertEqual(pos.n_sequences, 1)
        finally:
            handle.remove()
            pos._handle.remove()

    def test_from_last_needs_no_marker_but_response_does(self):
        PositionalSteering(None, "from_last")
        with self.assertRaises(ValueError):
            PositionalSteering(None, "response")

    def test_zero_and_random_constants(self):
        data = {"layer": 1, "constants": {"adapter_seed0": self.c.clone()}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "constants.pt"
            torch.save(data, path)
            loaded = load_constants(path)
        torch.testing.assert_close(get_constant(loaded, "adapter_seed0"), self.c)
        self.assertEqual(get_constant(loaded, "zero").abs().sum().item(), 0.0)
        r0, r1 = random_matched_vector(self.c, 0), random_matched_vector(self.c, 1)
        self.assertAlmostEqual(r0.norm().item(), self.c.norm().item(), places=5)
        self.assertFalse(torch.allclose(r0, r1))
        with self.assertRaises(KeyError):
            get_constant(loaded, "missing")

    def test_make_constants_measures_mean_and_spread(self):
        root = Path(__file__).resolve().parents[1]
        candidates = [root / "experiments/soo/scripts/make_constants.py", root / "make_constants.py"]  # repo, frozen snapshot
        script = next(c for c in candidates if c.exists())
        spec = importlib.util.spec_from_file_location("make_constants", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vecs = mod.collect(self.model, 1, [self.batch])
        self.assertEqual(tuple(vecs.shape), (2, 16))
        torch.testing.assert_close(vecs[0], self.base[0, 3])
        torch.testing.assert_close(vecs[1], self.base[1, 2])
        stats = mod.describe(vecs)
        self.assertGreater(stats["rms_spread"], 0.0)
        # A replaced model is constant: spread vanishes.
        with apply_steering(self.model, 1, self.c, 1.0, "replace"):
            const = mod.collect(self.model, 1, [self.batch])
        self.assertAlmostEqual(mod.describe(const)["rms_spread"], 0.0, places=6)
        torch.testing.assert_close(const.mean(0), self.c)

    def test_hooks_are_removed(self):
        with apply_steering(self.model, 1, self.c, 1.0, "replace", PositionalSteering(None, "from_last")):
            pass
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
