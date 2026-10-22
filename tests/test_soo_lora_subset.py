"""Software checks for restricting a LoRA adapter to a subset of layers/modules (LAYER_ROUND.md).

Run: .venv/bin/python -m unittest discover -s tests -p 'test_soo_lora_subset.py' -v
Tiny random Llama with a random LoRA on CPU; no model download. These check the
module swap, not any experimental result.
"""

import unittest

import torch

from selfconcept.soo.lora_subset import apply_adapter_subset, parse_layer_spec, parse_module_spec, restrict_lora


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=2, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return LlamaForCausalLM(config).cpu().eval()


def lora_model():
    from peft import LoraConfig, get_peft_model
    model = tiny_model()
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False))
    torch.manual_seed(11)
    for name, p in peft.named_parameters():
        if "lora_" in name:
            p.data.normal_(0, 0.5)
    return peft.eval()


def zero_lora_b(peft, layers):
    for name, p in peft.named_parameters():
        if "lora_B" in name and any(f".layers.{l}." in name for l in layers):
            p.data.zero_()


class LayerSpecTests(unittest.TestCase):
    def test_specs(self):
        self.assertEqual(parse_layer_spec("all", 4), {0, 1, 2, 3})
        self.assertEqual(parse_layer_spec("only:2", 4), {2})
        self.assertEqual(parse_layer_spec("except:2", 4), {0, 1, 3})
        self.assertEqual(parse_layer_spec("below:2", 4), {0, 1})
        self.assertEqual(parse_layer_spec("above:2", 4), {3})
        self.assertEqual(parse_layer_spec("range:1-2", 4), {1, 2})
        self.assertEqual(parse_layer_spec("layers:0,3", 4), {0, 3})
        for bad in ("only:4", "range:3-1", "layers:9", "foo:1", "only"):
            with self.assertRaises(ValueError):
                parse_layer_spec(bad, 4)
        self.assertEqual(parse_module_spec(None), ("q_proj", "v_proj"))
        self.assertEqual(parse_module_spec("q_proj"), ("q_proj",))


class RestrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 3], [1, 6, 9, 2, 7]]),
                      "attention_mask": torch.ones(2, 5, dtype=torch.long)}

    def logits(self, model):
        with torch.no_grad():
            return model(**self.batch).logits

    def test_all_layers_kept_leaves_the_adapter_untouched(self):
        peft = lora_model()
        full = self.logits(peft)
        stats = restrict_lora(peft, {0, 1, 2})
        self.assertEqual((stats["found"], stats["kept"], stats["removed"]), (6, 6, 0))
        torch.testing.assert_close(self.logits(peft), full)

    def test_only_one_layer_equals_zeroing_the_other_layers_deltas(self):
        reference = lora_model()
        zero_lora_b(reference, [0, 2])
        peft = lora_model()
        stats = restrict_lora(peft, {1})
        self.assertEqual((stats["kept"], stats["removed"], stats["kept_layers"]), (2, 4, [1]))
        torch.testing.assert_close(self.logits(peft), self.logits(reference))
        from peft.tuners.lora.layer import LoraLayer
        lora_names = [n for n, m in peft.named_modules() if isinstance(m, LoraLayer)]
        self.assertTrue(all(".layers.1." in n for n in lora_names) and len(lora_names) == 2)

    def test_restricted_model_differs_from_both_base_and_full(self):
        base = self.logits(tiny_model())
        peft = lora_model()
        full = self.logits(peft)
        restrict_lora(peft, {1})
        part = self.logits(peft)
        self.assertFalse(torch.allclose(part, base))
        self.assertFalse(torch.allclose(part, full))

    def test_module_filter_keeps_only_q_proj(self):
        reference = lora_model()
        for name, p in reference.named_parameters():
            if "lora_B" in name and "v_proj" in name:
                p.data.zero_()
        peft = lora_model()
        stats = apply_adapter_subset(peft, None, "q_proj")
        self.assertEqual((stats["kept"], stats["removed"], stats["kept_modules"]), (3, 3, ["q_proj"]))
        torch.testing.assert_close(self.logits(peft), self.logits(reference))

    def test_apply_adapter_subset_none_when_unset_and_unload_restores_base(self):
        peft = lora_model()
        self.assertIsNone(apply_adapter_subset(peft, None, None))
        stats = apply_adapter_subset(peft, "except:1", None)
        self.assertEqual(stats["kept_layers"], [0, 2])
        base = peft.unload().eval()
        torch.testing.assert_close(self.logits(base), self.logits(tiny_model()))
