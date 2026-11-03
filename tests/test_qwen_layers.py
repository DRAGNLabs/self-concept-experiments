"""Qwen3.5/3.8 keeps its decoder layers inside the multimodal wrapper."""
import unittest

from torch import nn

from selfconcept.assistant_axis.internals.model_specifics.qwen import QwenModelSpecifics


class _TextTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _TextTower()


class _Qwen35Wrapper(nn.Module):
    """Qwen3_5ForConditionalGeneration shape: model.language_model.layers, no model.layers."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()


class QwenLayerLookupTest(unittest.TestCase):
    def test_finds_text_tower_under_the_multimodal_wrapper(self):
        wrapper = _Qwen35Wrapper()
        layers = QwenModelSpecifics().get_layers(wrapper)
        self.assertIs(layers, wrapper.model.language_model.layers)
        self.assertEqual(len(layers), 4)
