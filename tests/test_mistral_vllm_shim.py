"""The Transformers 5.17 names vLLM 0.28's Pixtral wrapper still imports."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

from selfconcept.measurement.mistral_vllm_shim import patch_pixtral_module

_SITECUSTOMIZE = (
    Path(__file__).resolve().parents[1] / "experiments/deceiver-axis/vllm_shim/sitecustomize.py"
)


class PixtralShim(unittest.TestCase):
    def test_renamed_class_and_removed_helper_are_filled_in(self):
        module = types.ModuleType("pixtral")

        class PixtralVisionRotaryEmbedding:
            pass

        module.PixtralVisionRotaryEmbedding = PixtralVisionRotaryEmbedding
        patch_pixtral_module(module)
        self.assertIs(module.PixtralRotaryEmbedding, PixtralVisionRotaryEmbedding)
        with self.assertRaises(RuntimeError):
            module.position_ids_in_meshgrid([], max_width=1)

    def test_existing_names_are_left_alone(self):
        module = types.ModuleType("pixtral")
        module.PixtralRotaryEmbedding = "kept"
        module.PixtralVisionRotaryEmbedding = "new"
        module.position_ids_in_meshgrid = "kept-helper"
        patch_pixtral_module(module)
        self.assertEqual(module.PixtralRotaryEmbedding, "kept")
        self.assertEqual(module.position_ids_in_meshgrid, "kept-helper")

    def test_sitecustomize_points_at_the_shim(self):
        spec = importlib.util.spec_from_file_location("ms4_sitecustomize_check", _SITECUSTOMIZE)
        self.assertIsNotNone(spec)
        source = _SITECUSTOMIZE.read_text()
        shim = Path(__file__).resolve().parents[1] / "src/selfconcept/measurement/mistral_vllm_shim.py"
        self.assertIn("mistral_vllm_shim.py", source)
        self.assertTrue(shim.is_file())
        self.assertNotIn("mistral_vllm_shim", sys.modules)


if __name__ == "__main__":
    unittest.main()
