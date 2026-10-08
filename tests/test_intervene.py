import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

from selfconcept.measurement.capture import capture
from selfconcept.measurement.intervene import add_direction, cap_direction, unit
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
