"""Software checks for LoRA subsets on a hybrid (Gated DeltaNet + full attention) model.

Run: .venv-313/bin/python -m unittest discover -s tests -p 'test_soo_hybrid_subset.py' -v
A tiny random Qwen3.5 text model on CPU with LoRA on q_proj / v_proj (full-attention
layers) and the fused in_proj_qkv (linear-attention layers), as in the Qwen3.8-27B
adapters. These check the subset code, not any experimental result.
"""

import unittest

import torch

from selfconcept.soo.activations import attn_out_proj, get_decoder_layers
from selfconcept.soo.lora_subset import apply_adapter_subset, fused_qkv_slices

TARGETS = r"model\.layers\.\d+\.(self_attn\.(q_proj|v_proj)|linear_attn\.in_proj_qkv)"


def tiny_hybrid():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(3)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                               linear_conv_kernel_dim=4, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return Qwen3_5ForCausalLM(config).float().eval()


def lora(model):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(5)
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=TARGETS, init_lora_weights=False))
    return peft.eval()


def capture(model, module, batch):
    store = []
    h = module.register_forward_hook(lambda _m, _i, o: store.append(o.detach().clone()))
    try:
        with torch.no_grad():
            model(**batch)
    finally:
        h.remove()
    return store[-1]


class HybridSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 9, 7, 3]]), "attention_mask": torch.ones(1, 6, dtype=torch.long)}

    def qkv(self, model, layer=0):
        mod = get_decoder_layers(model)[layer].linear_attn.in_proj_qkv
        return capture(model, mod, self.batch)

    def test_slices_follow_the_query_key_value_layout(self):
        block = get_decoder_layers(tiny_hybrid())[0].linear_attn
        s = fused_qkv_slices(block)
        self.assertEqual(s, {"q": (0, 16), "k": (16, 32), "v": (32, 64)})
        self.assertEqual(block.in_proj_qkv.out_features, 64)

    def test_default_keeps_every_module_type(self):
        peft = lora(tiny_hybrid())
        stats = apply_adapter_subset(peft, "all", None)
        self.assertEqual(stats["removed"], 0)
        self.assertEqual(stats["kept_modules"], ["in_proj_qkv", "q_proj", "v_proj"])
        self.assertEqual(stats["module_spec"], "all")

    def test_value_slice_changes_only_value_channels(self):
        base = self.qkv(tiny_hybrid())
        full = self.qkv(lora(tiny_hybrid()))
        peft = lora(tiny_hybrid())
        stats = apply_adapter_subset(peft, "only:0", "in_proj_qkv.v")
        self.assertEqual((stats["kept"], stats["sliced"]), (1, 1))
        self.assertEqual(stats["kept_modules"], ["in_proj_qkv.v"])
        cut = self.qkv(peft)
        torch.testing.assert_close(cut[..., :32], base[..., :32])  # query and key untouched
        torch.testing.assert_close(cut[..., 32:], full[..., 32:])  # value channels carry the full delta
        self.assertGreater((full[..., :32] - base[..., :32]).abs().max().item(), 1e-4)  # the full adapter does touch q/k

    def test_query_key_slice_is_the_complement(self):
        base = self.qkv(tiny_hybrid())
        full = self.qkv(lora(tiny_hybrid()))
        peft = lora(tiny_hybrid())
        apply_adapter_subset(peft, "only:0", "in_proj_qkv.qk")
        cut = self.qkv(peft)
        torch.testing.assert_close(cut[..., :32], full[..., :32])
        torch.testing.assert_close(cut[..., 32:], base[..., 32:])

    def test_module_filters_select_layer_types(self):
        peft = lora(tiny_hybrid())
        stats = apply_adapter_subset(peft, "below:3", "in_proj_qkv")
        self.assertEqual((stats["kept"], stats["kept_layers"]), (3, [0, 1, 2]))
        peft = lora(tiny_hybrid())
        stats = apply_adapter_subset(peft, "all", "q_proj,v_proj")
        self.assertEqual((stats["kept"], stats["kept_layers"]), (2, [3]))
        peft = lora(tiny_hybrid())
        with self.assertRaises(ValueError):
            apply_adapter_subset(peft, "all", "v_proj.v")  # slices exist only on the fused projection

    def test_linear_attention_output_site_is_found(self):
        layers = get_decoder_layers(tiny_hybrid())
        self.assertIs(attn_out_proj(layers[0]), layers[0].linear_attn.out_proj)
        self.assertIs(attn_out_proj(layers[3]), layers[3].self_attn.o_proj)


if __name__ == "__main__":
    unittest.main()
