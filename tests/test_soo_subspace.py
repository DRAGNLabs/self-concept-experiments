"""Subspace math, intervention hooks, and a complete CPU pilot with random weights."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from selfconcept.soo.measure_overlap import main
from selfconcept.soo.overlap import measure_condition
from selfconcept.soo.steering import apply_steering
from selfconcept.soo.subspace import apply_subspace, fit_subspace, random_subspace
from test_soo_overlap import tiny_model, example_batch, pair_rows


class SubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uncentered_fit_keeps_mean_and_cancelling_differences(self):
        x = torch.tensor([[10., 1., 0.], [10., -1., 0.]])
        fitted = fit_subspace(x, 2)
        torch.testing.assert_close(fitted["basis"][:, 0].abs(), torch.tensor([1., 0., 0.]))
        self.assertFalse(fitted["centered"])
        self.assertEqual(fitted["numerical_rank"], 2)
        self.assertAlmostEqual(float(fitted["captured_energy_fraction"][-1]), 1.)
        cancelling = fit_subspace(torch.tensor([[2., 0.], [-2., 0.]]), 8)
        self.assertEqual(cancelling["basis"].shape, (2, 1))
        self.assertEqual(float(cancelling["mean_difference"].norm()), 0.)

    def test_random_subspace_reproducible_and_orthonormal(self):
        q = random_subspace(16, 4, 1)
        torch.testing.assert_close(q.T @ q, torch.eye(4), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(q, random_subspace(16, 4, 1), atol=0, rtol=0)
        self.assertFalse(torch.allclose(q, random_subspace(16, 4, 2)))
        with self.assertRaises(ValueError):
            fit_subspace(torch.zeros(2, 3), 2)

    def test_full_fractional_and_inactive_projection(self):
        model, batch = tiny_model(), example_batch()
        base = measure_condition(model, [batch], 1)
        q = random_subspace(16, 4, 5)
        for strength in (0., 0.5, 1.):
            result = measure_condition(model, [batch], 1,
                intervention=lambda: apply_subspace(model, 1, q, strength))
            before, after = result["hook_before"], result["hook_after"]
            torch.testing.assert_close(after, before - strength * ((before @ q) @ q.T))
            diff = before[:, 0] - before[:, 1]
            expected = diff.square().sum(-1) - (2 * strength - strength**2) * (diff @ q).square().sum(-1)
            actual = (after[:, 0] - after[:, 1]).square().sum(-1)
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)
            if strength == 0:
                for site in base:
                    torch.testing.assert_close(result[site], base[site], rtol=0, atol=0)
            if strength == 1:
                torch.testing.assert_close(after @ q, torch.zeros(2, 2, 4), atol=1e-8, rtol=0)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_rank_one_matches_existing_projection(self):
        model, batch = tiny_model(), example_batch()
        vector = torch.arange(1., 17.)
        unit = (vector / vector.norm())[:, None]
        old = measure_condition(model, [batch], 1,
            intervention=lambda: apply_steering(model, 1, vector, 1., "project"))
        new = measure_condition(model, [batch], 1,
            intervention=lambda: apply_subspace(model, 1, unit, 1.))
        for site in old:
            torch.testing.assert_close(old[site], new[site], atol=1e-6, rtol=1e-5)

    def test_bad_basis_rejected_without_hook_leak(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            with apply_subspace(model, 1, torch.ones(16, 2), 1.):
                pass
        with self.assertRaises(ValueError):
            with apply_subspace(model, 1, random_subspace(16, 2, 0), 2.):
                pass
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_complete_pilot_fits_only_declared_fit_rows(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            tiny_model().save_pretrained(model_path)
            vocab = {"[PAD]": 0, "[EOS]": 2, "[UNK]": 3, "self": 4,
                     "other": 5, "longer": 6, "0": 7, "1": 8, "assistant": 9}
            backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
            tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
            tokenizer.save_pretrained(model_path)
            rows = pair_rows()
            for i in range(2):
                for kind in ("self_other", "nonsocial"):
                    rows.append(dict(id=f"fit_{kind}_{i}", family="fit_family", split="fit", kind=kind,
                                     self_prompt=f"self {i} longer" if kind == "self_other" else f"longer other {i}",
                                     other_prompt=f"other {i}" if kind == "self_other" else f"{i} other longer"))
            pairs = root / "pairs.jsonl"
            pairs.write_text("".join(json.dumps(row) + "\n" for row in rows))
            out = root / "run"
            main(["--model", str(model_path), "--probes", str(pairs), "--out", str(out),
                  "--layer", "1", "--subspace-ranks", "1", "2", "4",
                  "--subspace-strengths", "0", "1", "--random-seeds", "0", "1", "2",
                  "--bootstrap", "100", "--checkpoint-conditions"])
            fit = torch.load(out / "subspace_fit.pt", weights_only=True)
            self.assertEqual(len(fit["pairs"]), 4)
            self.assertTrue(all(p["split"] == "fit" for p in fit["pairs"]))
            self.assertEqual(fit["skipped_ranks"], [4])
            self.assertEqual(fit["fits"]["self_other"]["n_pairs"], 2)
            conditions = torch.load(out / "activations.pt", weights_only=True)
            self.assertEqual(len(conditions), 13)
            self.assertEqual(len(list((out / "conditions").glob("*.pt"))), 13)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(set(manifest["condition_settings"]), set(conditions))
            self.assertTrue(all(p["split"] == "development" for p in manifest["pairs"]))
            self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "complete")
            self.assertEqual(conditions["base"]["hook_before"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
