"""Measurement checks on controlled tensors and a tiny random causal LM.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_soo_overlap.py' -v
No model download or GPU is required. These are software checks, not SOO results.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

import torch

from selfconcept.soo.activations import attn_out_proj, get_decoder_layers
from selfconcept.soo.measure_overlap import main, read_pairs
from selfconcept.soo.overlap import (
    endpoint, last_valid_indices, measure_condition, paired_interval, site_metrics, summarize,
)
from selfconcept.soo.steering import PositionalSteering, apply_steering


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=2, attention_dropout=0.3,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return LlamaForCausalLM(config).cpu()


def example_batch():
    return {
        "input_ids": torch.tensor([[1, 4, 5, 9, 0], [1, 6, 9, 0, 0],
                                   [1, 7, 6, 5, 9], [1, 8, 9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
    }


def pair_rows():
    return [dict(id=str(i), family=f"family_{i}", split="development", kind="self_other",
                 self_prompt=f"self {i}", other_prompt=f"other longer {i}") for i in range(2)]


class MetricsTests(unittest.TestCase):
    def test_endpoint_both_padding_sides_and_empty(self):
        values = torch.arange(24).reshape(2, 4, 3)
        right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        torch.testing.assert_close(endpoint(values, right), values[[0, 1], [1, 2]].float())
        torch.testing.assert_close(endpoint((values,), left), values[:, -1].float())
        with self.assertRaises(ValueError):
            last_valid_indices(torch.tensor([[0, 0]]))

    def test_gap_mean_separation_and_collapse_are_distinct(self):
        x = torch.tensor([[[1., 0.], [-1., 0.]], [[-1., 0.], [1., 0.]]])
        report, _ = site_metrics(x)
        self.assertEqual(report["gap"], 2.)
        self.assertEqual(report["centroid_gap"], 0.)
        self.assertGreater(report["prompt_variance"], 0.)
        collapsed, _ = site_metrics(torch.zeros_like(x))
        self.assertEqual(collapsed["gap"], 0.)
        self.assertEqual(collapsed["prompt_variance"], 0.)

    def test_family_bootstrap_not_independent_pair_bootstrap(self):
        self.assertIsNone(paired_interval([1., 2.], ["same", "same"]))
        interval = paired_interval([0., 0., 10.], ["a", "a", "b"], 1000)
        self.assertEqual(interval, [0., 10.])
        self.assertEqual(paired_interval([2., 2.], ["a", "b"]), [2., 2.])

    def test_split_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            def write(rows):
                path.write_text("".join(json.dumps(p) + "\n" for p in rows))
            rows = pair_rows()
            write(rows)
            self.assertEqual(len(read_pairs(path, "development")[1]), 2)
            for invalid in (
                [rows[0], rows[0]],
                [rows[0], {**rows[1], "split": "final", "family": rows[0]["family"]}],
                [rows[0], {**rows[1], "split": "fit", "self_prompt": rows[0]["self_prompt"]}],
                [{**rows[0], "family": ""}],
            ):
                write(invalid)
                with self.assertRaises(ValueError):
                    read_pairs(path, "development")

    def test_paired_reports_separate_controls(self):
        rows = pair_rows()
        rows[1]["kind"] = "nonsocial"
        base = torch.tensor([[[2., 0.], [0., 0.]], [[4., 0.], [0., 0.]]])
        summary, records = summarize({"base": {"site": base}, "half": {"site": base / 2}}, rows)
        self.assertEqual(summary["half"]["self_other"]["site"]["gap_change_vs_base"], -1.5)
        self.assertEqual(summary["half"]["nonsocial"]["site"]["gap_change_vs_base"], -6.)
        self.assertEqual(len(records), 4)


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = tiny_model()
        self.batch = example_batch()
        self.vector = torch.linspace(-0.1, 0.2, 16)
        self.base = measure_condition(self.model, [self.batch], 1)

    def run_steer(self, mode, alpha, positions=None):
        return measure_condition(self.model, [self.batch], 1,
                                 intervention=lambda: apply_steering(
                                     self.model, 1, self.vector, alpha, mode, positions))

    def test_base_capture_matches_model_hidden_states(self):
        self.assertFalse(self.model.training)
        with torch.inference_mode():
            hidden = self.model(**self.batch, output_hidden_states=True).hidden_states[-1]
        expected = endpoint(hidden, self.batch["attention_mask"]).reshape(2, 2, 16)
        torch.testing.assert_close(self.base["final_norm"], expected)
        torch.testing.assert_close(self.base["hook_before"], self.base["hook_after"], rtol=0, atol=0)

    def test_inactive_hooks_and_cleanup(self):
        for mode in ("add", "project"):
            inactive = self.run_steer(mode, 0.)
            for site in self.base:
                torch.testing.assert_close(inactive[site], self.base[site], rtol=0, atol=0)
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_fixed_translation_identity_and_downstream_change(self):
        result = self.run_steer("add", 2.)
        before, after = result["hook_before"], result["hook_after"]
        torch.testing.assert_close(before, self.base["hook_before"], rtol=0, atol=0)
        torch.testing.assert_close(after, before - 2 * self.vector)
        torch.testing.assert_close(after[:, 0] - after[:, 1], before[:, 0] - before[:, 1], atol=1e-7, rtol=1e-5)
        self.assertFalse(torch.allclose(result["final_norm"], self.base["final_norm"]))
        # The local identity must not be enforced as a false downstream identity.
        base_gap = site_metrics(self.base["final_norm"])[0]["gap"]
        steered_gap = site_metrics(result["final_norm"])[0]["gap"]
        self.assertGreater(abs(base_gap - steered_gap), 1e-6)

    def test_projection_removes_component_and_contracts_local_gap(self):
        result = self.run_steer("project", 1.)
        u = self.vector / self.vector.norm()
        after = result["hook_after"]
        torch.testing.assert_close(after @ u, torch.zeros(2, 2), atol=1e-8, rtol=0)
        before_gap = (result["hook_before"][:, 0] - result["hook_before"][:, 1]).square().sum(-1)
        after_gap = (after[:, 0] - after[:, 1]).square().sum(-1)
        self.assertTrue((after_gap <= before_gap + 1e-9).all())

    def test_inactive_gate_and_asymmetric_shift(self):
        # Test the generic hook interface only; this is not a trained probe gate.
        @contextmanager
        def gated(scores):
            module = attn_out_proj(get_decoder_layers(self.model)[1])
            handle = module.register_forward_hook(
                lambda _m, _i, out: out - scores[:, None, None] * self.vector)
            try:
                yield
            finally:
                handle.remove()
        inactive = measure_condition(self.model, [self.batch], 1,
                                     intervention=lambda: gated(torch.zeros(4)))
        for site in self.base:
            torch.testing.assert_close(inactive[site], self.base[site], rtol=0, atol=0)
        active = measure_condition(self.model, [self.batch], 1,
                                   intervention=lambda: gated(torch.tensor([1., 0., 1., 0.])))
        before, after = active["hook_before"], active["hook_after"]
        torch.testing.assert_close(after[:, 0] - after[:, 1], before[:, 0] - before[:, 1] - self.vector)

    def test_padding_does_not_change_measurements(self):
        # Vary padding amount and batch composition while preserving valid tokens.
        singles = []
        for start in (0, 2):
            batch = {k: v[start:start + 2] for k, v in self.batch.items()}
            width = int(batch["attention_mask"].sum(1).max())
            singles.append({k: v[:, :width] for k, v in batch.items()})
        separate = measure_condition(self.model, singles, 1)
        for site in self.base:
            torch.testing.assert_close(separate[site], self.base[site], atol=1e-6, rtol=1e-5)

    def test_position_mask_matches_response_header(self):
        prompt = self.run_steer("add", 1., PositionalSteering([9], "prompt"))
        response = self.run_steer("add", 1., PositionalSteering([9], "response"))
        # Every endpoint is the one-token assistant header, excluded by prompt-only.
        torch.testing.assert_close(prompt["hook_after"], prompt["hook_before"], rtol=0, atol=0)
        torch.testing.assert_close(response["hook_after"], response["hook_before"] - self.vector)
        # Changing earlier positions still changes later processing of the endpoint.
        self.assertFalse(torch.allclose(prompt["final_norm"], self.base["final_norm"]))

    def test_cleanup_when_forward_raises(self):
        broken = {**self.batch, "input_ids": torch.full_like(self.batch["input_ids"], 100)}
        with self.assertRaises((IndexError, RuntimeError)):
            measure_condition(self.model, [broken], 1,
                              intervention=lambda: apply_steering(self.model, 1, self.vector, 1.))
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)

    def test_qwen_hybrid_wrapper_capture_sites(self):
        # Same architecture family as the first planned experimental model;
        # tiny random weights exercise both linear and full attention sites.
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        config = Qwen3_5Config(
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
                             linear_num_key_heads=2, linear_num_value_heads=2,
                             layer_types=["linear_attention", "full_attention"],
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                              "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
                             pad_token_id=0),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
                               out_hidden_size=16, patch_size=2, spatial_merge_size=1,
                               temporal_patch_size=1, num_position_embeddings=16),
            image_token_id=29, video_token_id=30, vision_start_token_id=31,
        )
        model = Qwen3_5ForConditionalGeneration(config).cpu()
        for layer in (0, 1):
            captured = measure_condition(model, [self.batch], layer,
                                         intervention=lambda: apply_steering(model, layer, self.vector, 1., "project"))
            torch.testing.assert_close(captured["hook_after"] @ (self.vector / self.vector.norm()),
                                       torch.zeros(2, 2), atol=1e-7, rtol=0)
            self.assertEqual(captured["final_norm"].shape, (2, 2, 16))

    def test_local_cli_with_adapter_and_saved_provenance(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from peft import LoraConfig, get_peft_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            self.model.save_pretrained(model_path)
            vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                     "self": 4, "other": 5, "longer": 6, "0": 7, "1": 8, "assistant": 9}
            backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
            tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
            tokenizer.save_pretrained(model_path)
            rows = pair_rows()
            probes = root / "probes.jsonl"
            probes.write_text("".join(json.dumps(row) + "\n" for row in rows))
            vectors = root / "vectors.pt"
            torch.save({"model": str(model_path), "vectors_last": self.vector.repeat(3, 1)}, vectors)
            adapter = get_peft_model(self.model, LoraConfig(r=2, lora_alpha=2, target_modules=["o_proj"], lora_dropout=0.5))
            with torch.no_grad():
                for name, param in adapter.named_parameters():
                    if "lora_B" in name:
                        param.fill_(0.2)
            adapter.save_pretrained(root / "adapter")
            output = root / "run"
            args = ["--model", str(model_path), "--probes", str(probes), "--out", str(output),
                    "--layer", "1", "--vectors", str(vectors), "--adapter", str(root / "adapter"),
                    "--mode", "project", "--random-seeds", "0", "1", "2", "--bootstrap", "100"]
            main(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["declared_split_ids"]["development"], ["0", "1"])
            self.assertIn(str(probes), manifest["input_sha256"])
            self.assertIn(str(root / "adapter/adapter_model.safetensors"), manifest["input_sha256"])
            self.assertIn(str(model_path / "model.safetensors"), manifest["input_sha256"])
            activations = torch.load(output / "activations.pt", weights_only=True)
            self.assertEqual(len(activations), 7)
            self.assertFalse(torch.allclose(activations["base"]["hook_before"], activations["adapter"]["hook_before"]))
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved["inactive"]["self_other"]["hook_after"]["gap_change_vs_base"], 0.)
            tokens = [json.loads(line) for line in (output / "tokens.jsonl").read_text().splitlines()]
            self.assertTrue(all(t["endpoint_token"] == "assistant" for t in tokens))
            with self.assertRaises(ValueError):
                main(args)


if __name__ == "__main__":
    unittest.main()
