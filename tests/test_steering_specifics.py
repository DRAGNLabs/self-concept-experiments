"""CPU tests for selfconcept.measurement.steering_specifics: name resolution, the completion splitters (including
Qwen's thinking-off case), the residual-stream readers and the vLLM class patch for both forward contracts."""

import unittest

import torch

from selfconcept.measurement.steering_specifics import (
    Gemma4SteeringSpecifics, GptOssSteeringSpecifics, MuseGlimmerSteeringSpecifics, Qwen35SteeringSpecifics,
    split_answer_only, steering_model_specifics)


class Resolution(unittest.TestCase):
    def test_known_models(self):
        cases = {
            "openai/gpt-oss-120b": (GptOssSteeringSpecifics, "gpt-oss"),
            "google/gemma-4-31B-it": (Gemma4SteeringSpecifics, "gemma4"),
            "meta-models/Muse-Glimmer-30B": (MuseGlimmerSteeringSpecifics, "muse-glimmer"),
            "Qwen/Qwen3.8-27B": (Qwen35SteeringSpecifics, "qwen3.5"),
            "Qwen/Qwen3.5-27B": (Qwen35SteeringSpecifics, "qwen3.5"),
        }
        for name, (cls, family) in cases.items():
            with self.subTest(name=name):
                specifics = steering_model_specifics(name)
                self.assertIsInstance(specifics, cls)
                self.assertEqual(specifics.family, family)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            steering_model_specifics("meta-llama/Llama-3.1-8B-Instruct")

    def test_only_gpt_oss_has_an_hf_attention_kernel(self):
        self.assertEqual(GptOssSteeringSpecifics().hf_attn_implementation, "kernels-community/vllm-flash-attn3")
        for specifics in (Gemma4SteeringSpecifics(), MuseGlimmerSteeringSpecifics(), Qwen35SteeringSpecifics()):
            self.assertIsNone(specifics.hf_attn_implementation)


class Splitters(unittest.TestCase):
    def test_gemma_and_muse_reasoning_is_a_string_not_none(self):
        self.assertEqual(Gemma4SteeringSpecifics().split_completion("just an answer"), ("", "just an answer"))
        self.assertEqual(MuseGlimmerSteeringSpecifics().split_completion("just an answer"), ("", "just an answer"))

    def test_muse_channels(self):
        raw = "<|start|>assistant to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>answer<|eot|>"
        self.assertEqual(MuseGlimmerSteeringSpecifics().split_completion(raw), ("think", "answer"))

    def test_qwen_thinking_on_and_off(self):
        qwen = Qwen35SteeringSpecifics()
        self.assertEqual(qwen.completion_splitter({})("a</think>b"), ("a", "b"))
        self.assertEqual(qwen.completion_splitter({"enable_thinking": True})("a</think>b"), ("a", "b"))
        self.assertIs(qwen.completion_splitter({"enable_thinking": False}), split_answer_only)
        self.assertEqual(split_answer_only("  the answer\n"), ("", "the answer"))

    def test_fixed_splitters_ignore_chat_kwargs(self):
        for specifics in (GptOssSteeringSpecifics(), Gemma4SteeringSpecifics(), MuseGlimmerSteeringSpecifics()):
            self.assertEqual(specifics.completion_splitter({"enable_thinking": False}), specifics.split_completion)


class _SumBlock:
    """gpt-oss / Qwen3.5 contract: forward returns (mlp_output, residual)."""
    layer_idx = 1

    def forward(self, hidden_states, residual):
        return hidden_states * 2, residual


class _StreamLayer:
    """Gemma 4 / Muse contract: forward returns (stream, other)."""

    def __init__(self):
        self.self_attn = type("Attn", (), {"layer_idx": 1})()

    def forward(self, hidden_states):
        return hidden_states * 2, None


class StreamsAndPatches(unittest.TestCase):
    def test_stream_readers(self):
        a, b = torch.ones(2, 3), torch.full((2, 3), 2.0)
        for specifics in (GptOssSteeringSpecifics(), Qwen35SteeringSpecifics()):
            torch.testing.assert_close(specifics.vllm_stream((a, b)), a + b)
        for specifics in (Gemma4SteeringSpecifics(), MuseGlimmerSteeringSpecifics()):
            torch.testing.assert_close(specifics.vllm_stream((a, b)), a)

    def test_patch_sum_contract(self):
        for specifics in (GptOssSteeringSpecifics(), Qwen35SteeringSpecifics()):
            cls = type("Block", (_SumBlock,), {})
            restore = specifics.patch_vllm(cls, {1: lambda stream: stream + 10})
            mlp, residual = cls().forward(torch.ones(2), torch.ones(2))
            torch.testing.assert_close(mlp + residual, torch.full((2,), 13.0))  # stream 3 -> 13
            torch.testing.assert_close(residual, torch.ones(2))
            restore()
            torch.testing.assert_close(cls().forward(torch.ones(2), torch.ones(2))[0], torch.full((2,), 2.0))

    def test_patch_stream_contract(self):
        for specifics in (Gemma4SteeringSpecifics(), MuseGlimmerSteeringSpecifics()):
            cls = type("Layer", (_StreamLayer,), {})
            restore = specifics.patch_vllm(cls, {1: lambda stream: stream + 10})
            stream, other = cls().forward(torch.ones(2))
            torch.testing.assert_close(stream, torch.full((2,), 12.0))
            self.assertIsNone(other)
            restore()
            torch.testing.assert_close(cls().forward(torch.ones(2))[0], torch.full((2,), 2.0))

    def test_unlisted_layer_passes_through(self):
        cls = type("Block", (_SumBlock,), {})
        restore = GptOssSteeringSpecifics().patch_vllm(cls, {0: lambda stream: stream + 10})
        torch.testing.assert_close(cls().forward(torch.ones(2), torch.ones(2))[0], torch.full((2,), 2.0))
        restore()


if __name__ == "__main__":
    unittest.main()
