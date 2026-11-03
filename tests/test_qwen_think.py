"""Qwen3.5/3.8 thinking-on support: the completion splitter, the stored turn and the vLLM block patch's call shape."""
import unittest


class SplitQwenThinkCompletionTest(unittest.TestCase):
    def test_splits_at_close_tag(self):
        from selfconcept.common.generation.model_specifics.think_flag import split_qwen_think_completion

        self.assertEqual(split_qwen_think_completion("plan it\n</think>\n\n```python\nx = 1\n```<|im_end|>"),
                         ("plan it", "```python\nx = 1\n```"))

    def test_no_close_tag_is_all_reasoning(self):
        from selfconcept.common.generation.model_specifics.think_flag import split_qwen_think_completion

        self.assertEqual(split_qwen_think_completion("still thinking about"), ("still thinking about", ""))

    def test_only_first_close_tag_splits(self):
        from selfconcept.common.generation.model_specifics.think_flag import split_qwen_think_completion

        self.assertEqual(split_qwen_think_completion("<think>a</think>b </think> c"), ("a", "b </think> c"))

    def test_assistant_turn_round_trip(self):
        from selfconcept.common.generation.model_specifics.think_tag import QwenThinkGenerationSpecifics

        specifics = QwenThinkGenerationSpecifics()
        turn = specifics.to_assistant_turn("r\n</think>\n\nanswer")
        self.assertEqual(turn, {"role": "assistant", "content": "answer", "reasoning_content": "r"})
        self.assertEqual(specifics.to_assistant_turn(specifics.to_completion(turn)), turn)
        self.assertEqual(specifics.to_assistant_turn("bare"), {"role": "assistant", "content": "bare"})


class QwenVllmPatchTest(unittest.TestCase):
    def test_keyword_call_shape(self):
        """Qwen3NextModel calls layer(positions=, hidden_states=, residual=) and the layer returns (mlp_output, residual)."""
        import torch
        from torch import nn
        from selfconcept.measurement.intervene import add_transform, patch_vllm_blocks

        class Layer(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

            def forward(self, hidden_states, residual, positions=None, **kwargs):
                return hidden_states * 3, (hidden_states if residual is None else residual)

        direction = torch.zeros(4); direction[2] = 1.0
        restore = patch_vllm_blocks(Layer, {1: add_transform(direction, 2.0)})
        h, r = torch.ones(3, 4), torch.full((3, 4), 0.5)
        out, res = Layer(1)(positions=None, hidden_states=h, residual=r)
        self.assertTrue(torch.allclose(out + res, h * 3 + r + 2.0 * direction))
        out0, res0 = Layer(0)(positions=None, hidden_states=h, residual=None)
        self.assertTrue(torch.equal(out0, h * 3) and torch.equal(res0, h))
        restore()


if __name__ == "__main__":
    unittest.main()
