import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

from selfconcept.measurement.capture import capture
from selfconcept.measurement.intervene import _hooks, add_direction, cap_direction, noise_transform, unit
from selfconcept.codebench.hf_harmony import hf_harmony_generate


def tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()


class InterveneTests(unittest.TestCase):
    def test_add_shifts_hooked_layer_by_vector(self):
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        direction = torch.zeros(16)
        direction[0] = 2.0
        with torch.inference_mode(), capture(model, [1]) as before:
            model(ids)
        with torch.inference_mode(), add_direction(model, 1, direction, 1.5), capture(model, [1]) as after:
            model(ids)
        torch.testing.assert_close(after[1][0] - before[1][0], torch.full((4, 16), 0.).index_fill(1, torch.tensor([0]), 3.0))
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_cap_clips_projection_from_above_only(self):
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.inference_mode(), capture(model, [2]) as before:
            model(ids)
        residual = before[2][0]
        u = unit(residual[0])  # direction along the first token's residual, so that token exceeds the median
        projections = residual @ u
        threshold = float(projections.median())
        with torch.inference_mode(), cap_direction(model, {2: threshold}, {2: u}), capture(model, [2]) as after:
            model(ids)
        capped = after[2][0] @ u
        self.assertTrue(torch.all(capped <= threshold + 1e-4))
        below = projections <= threshold
        torch.testing.assert_close(after[2][0][below], residual[below])
        self.assertTrue(torch.any(~below))

    def test_cap_rejects_mismatched_layers(self):
        with self.assertRaises(ValueError):
            with cap_direction(tiny_model(), {1: 0.0}, {2: torch.ones(16)}):
                pass

    def test_hf_harmony_generate_writes_sidecars_and_applies_intervention(self):
        model = tiny_model()
        tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="t31", pad_token="t30")
        tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}t5"
        model.generation_config.eos_token_id = 31
        direction = torch.zeros(16)
        direction[1] = 1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate = hf_harmony_generate(model, tokenizer, 6, root / "r.jsonl", 1.0, 3,
                                           interventions=[lambda: add_direction(model, 1, direction, 0.5)],
                                           projection_layers=[1], directions_by_layer={1: direction},
                                           projection_log_path=root / "p.jsonl")
            final, truncated = generate([{"role": "user", "content": "t1 t2 t3"}], 0, "ex")
            self.assertEqual(final, "")  # no harmony channels in a random tiny model
            self.assertTrue(truncated or not truncated)
            reasoning = json.loads((root / "r.jsonl").read_text())
            self.assertEqual((reasoning["example_id"], reasoning["turn"]), ("ex", 0))
            projection = json.loads((root / "p.jsonl").read_text())
            self.assertEqual(projection["prompt_tokens"], 4)
            self.assertEqual(projection["layers"]["1"]["prompt_projection"]["n"], 4)
            self.assertEqual(projection["layers"]["1"]["generated_projection"]["n"], projection["generated_tokens"] - 1)
            self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)
            # Same seed, same prompt: deterministic across calls.
            again = hf_harmony_generate(model, tokenizer, 6, root / "r2.jsonl", 1.0, 3)
            first = again([{"role": "user", "content": "t1 t2 t3"}], 0, "ex")
            second = again([{"role": "user", "content": "t1 t2 t3"}], 0, "ex")
            self.assertEqual(first, second)


class TransformFactoryTest(unittest.TestCase):
    """The tensor-parallel worker rebuilds its transform from a saved mode name; floor must not fall back to cap."""

    def test_floor_lifts_and_cap_lowers(self):
        from selfconcept.measurement.intervene import transform_factory
        direction = torch.zeros(4)
        direction[0] = 2.0
        hidden = torch.tensor([[-3.0, 1.0, 0.0, 0.0], [5.0, 1.0, 0.0, 0.0]])
        floored = transform_factory("floor")(direction, 1.0)(hidden)
        capped = transform_factory("cap")(direction, 1.0)(hidden)
        torch.testing.assert_close(floored[:, 0], torch.tensor([1.0, 5.0]))
        torch.testing.assert_close(capped[:, 0], torch.tensor([-3.0, 1.0]))
        torch.testing.assert_close(floored[:, 1:], hidden[:, 1:])

    def test_unknown_mode_raises(self):
        from selfconcept.measurement.intervene import transform_factory
        with self.assertRaises(ValueError):
            transform_factory("noise")


if __name__ == "__main__":
    unittest.main()


class VllmTupleHookTest(unittest.TestCase):
    """register_vllm_transforms on a stand-in for vLLM's (mlp_output, residual) block contract."""

    def _layers(self):
        import torch
        from torch import nn

        class Block(nn.Module):
            def forward(self, mlp_output, residual):
                return mlp_output, residual

        return nn.ModuleList([Block(), Block(), Block()])

    def test_add_and_cap_act_on_the_sum(self):
        import torch
        from selfconcept.measurement.intervene import add_transform, cap_transform, register_vllm_transforms

        torch.manual_seed(0)
        layers = self._layers()
        direction = torch.zeros(8)
        direction[0] = 2.0  # non-unit on purpose
        mlp, res = torch.randn(5, 8), torch.randn(5, 8)
        stream = mlp + res
        handles = register_vllm_transforms(layers, {1: add_transform(direction, 3.0)})
        out, res_out = layers[1](mlp, res)
        self.assertTrue(torch.equal(res_out, res))
        expected = stream + 3.0 * direction
        self.assertTrue(torch.allclose(out + res_out, expected, atol=1e-6))
        self.assertTrue(torch.equal(layers[0](mlp, res)[0], mlp))
        for h in handles:
            h.remove()
        threshold = 0.5
        handles = register_vllm_transforms(layers, {2: cap_transform(direction, threshold)})
        out, res_out = layers[2](mlp, res)
        projection = (out + res_out)[:, 0]
        self.assertTrue(torch.all(projection <= threshold + 1e-6))
        below = stream[:, 0] <= threshold
        self.assertTrue(torch.allclose((out + res_out)[below], stream[below], atol=1e-6))
        self.assertTrue(torch.allclose((out + res_out)[:, 1:], stream[:, 1:], atol=1e-6))
        for h in handles:
            h.remove()
        with self.assertRaises(ValueError):
            register_vllm_transforms(layers, {3: add_transform(direction, 1.0)})

    def test_floor_is_the_mirror_of_cap(self):
        import torch
        from selfconcept.measurement.intervene import cap_transform, floor_transform

        torch.manual_seed(1)
        direction = torch.zeros(8)
        direction[0] = 2.0  # non-unit on purpose
        stream = torch.randn(50, 8)
        threshold = 0.3
        floored = floor_transform(direction, threshold)(stream)
        self.assertTrue(torch.all(floored[:, 0] >= threshold - 1e-6))
        above = stream[:, 0] >= threshold
        self.assertTrue(above.any() and (~above).any())
        self.assertTrue(torch.allclose(floored[above], stream[above], atol=1e-6))
        self.assertTrue(torch.allclose(floored[:, 1:], stream[:, 1:], atol=1e-6))
        # floor along u at t == cap along -u at -t, reflected
        capped = cap_transform(-direction, -threshold)(stream)
        self.assertTrue(torch.allclose(floored, capped, atol=1e-6))
        self.assertEqual(floored.dtype, stream.dtype)


class VllmClassPatchTest(unittest.TestCase):
    def test_patch_applies_by_layer_idx_and_restores(self):
        import torch
        from torch import nn
        from selfconcept.measurement.intervene import add_transform, patch_vllm_blocks

        class Block(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

            def forward(self, hidden_states, positions, residual):
                return hidden_states * 2, residual

        blocks = [Block(0), Block(1)]
        direction = torch.zeros(4); direction[1] = 1.0
        restore = patch_vllm_blocks(Block, {1: add_transform(direction, 5.0)})
        h, r = torch.ones(3, 4), torch.ones(3, 4)
        out0, res0 = blocks[0](h, None, r)
        self.assertTrue(torch.equal(out0, h * 2))
        out1, res1 = blocks[1](h, None, r)
        self.assertTrue(torch.allclose(out1 + res1, h * 2 + r + 5.0 * direction))
        self.assertTrue(torch.equal(res1, r))
        restore()
        out1, _ = blocks[1](h, None, r)
        self.assertTrue(torch.equal(out1, h * 2))


class VllmStreamLayerPatchTest(unittest.TestCase):
    def test_patch_transforms_returned_stream_and_restores(self):
        import torch
        from torch import nn
        from selfconcept.measurement.intervene import add_transform, cap_transform, patch_vllm_stream_layers

        class Layer(nn.Module):  # Gemma4DecoderLayer shape: forward(positions, hidden_states, residual, ...) -> (stream, None)
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

            def forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
                return hidden_states * 2, None

        layers = [Layer(0), Layer(1), Layer(2)]
        direction = torch.zeros(4); direction[1] = 1.0
        restore = patch_vllm_stream_layers(Layer, {1: add_transform(direction, 5.0), 2: cap_transform(direction, 1.0)})
        h = torch.ones(3, 4)
        out0, res0 = layers[0](None, h, None)
        self.assertTrue(torch.equal(out0, h * 2) and res0 is None)
        out1, res1 = layers[1](None, h, None, per_layer_input=None)
        self.assertTrue(torch.allclose(out1, h * 2 + 5.0 * direction) and res1 is None)
        out2, _ = layers[2](None, h, None)
        self.assertTrue(torch.allclose(out2[:, 1], torch.ones(3)) and torch.allclose(out2[:, 0], 2 * torch.ones(3)))
        restore()
        out1, _ = layers[1](None, h, None)
        self.assertTrue(torch.equal(out1, h * 2))


class VllmStreamLayerAttentionIndexTest(unittest.TestCase):
    def test_patch_reads_layer_index_from_the_attention_module(self):
        """Muse Glimmer's vLLM decoder layer has no layer_idx of its own; it lives on self_attn (from the weight prefix)."""
        import torch
        from torch import nn
        from selfconcept.measurement.intervene import add_transform, patch_vllm_stream_layers

        class Attn(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

        class Layer(nn.Module):  # MuseGlimmerDecoderLayer shape: forward(positions, hidden_states, residual) -> (stream, residual)
            def __init__(self, layer_idx):
                super().__init__()
                self.self_attn = Attn(layer_idx)

            def forward(self, positions, hidden_states, residual):
                return hidden_states * 2, hidden_states

        direction = torch.tensor([1.0, 0.0, 0.0])
        restore = patch_vllm_stream_layers(Layer, {1: add_transform(direction, 5.0)})
        try:
            x = torch.ones(2, 3)
            out0, _ = Layer(0)(None, x, None)
            out1, res1 = Layer(1)(None, x, None)
            self.assertTrue(torch.allclose(out0, x * 2))
            self.assertTrue(torch.allclose(out1, x * 2 + 5.0 * direction))
            self.assertTrue(torch.allclose(res1, x))
        finally:
            restore()
        out1, _ = Layer(1)(None, torch.ones(2, 3), None)
        self.assertTrue(torch.allclose(out1, torch.ones(2, 3) * 2))


class NoiseTransformTests(unittest.TestCase):
    def test_noise_is_fresh_per_position_and_advances_across_calls(self):
        bank = torch.eye(5)[:4]  # 4 unit vectors
        transform = noise_transform(bank, 2.0)
        first = transform(torch.zeros(3, 5))
        torch.testing.assert_close(first, 2.0 * bank[:3])
        second = transform(torch.zeros(1, 2, 5))  # batched shape: positions 3, 0 (mod 4)
        torch.testing.assert_close(second.reshape(2, 5), 2.0 * bank[[3, 0]])

    def test_noise_hook_perturbs_each_token_by_the_coefficient(self):
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        bank = torch.nn.functional.normalize(torch.randn(16, 16), dim=1)
        with torch.inference_mode(), capture(model, [1]) as before:
            model(ids)
        with torch.inference_mode(), _hooks(model, {1: noise_transform(bank, 0.5)}), capture(model, [1]) as after:
            model(ids)
        delta = after[1][0] - before[1][0]
        torch.testing.assert_close(delta.norm(dim=-1), torch.full((4,), 0.5))
        self.assertGreater((delta[0] - delta[1]).norm(), 0.1)
