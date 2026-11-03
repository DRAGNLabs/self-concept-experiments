"""Integrity and scientific decision-rule regressions for the Part 3 bridge."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/soo/scripts'))
import part3_existing_audit as audit
import part3_band as band
import goal_identity_task as task
from test_soo_goal_identity import historical, oracle


class Part3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [e for split in band.SPLITS for e in task.build(split, historical(split))]

    def test_independent_keys_all_variants(self):
        for e in self.rows:
            k = audit.independent_keys(e)
            for field in ('expected_answer', 'assigned_goal_answer', 'recipient_optimal_answer', 'dominated_answer'):
                self.assertEqual(k[field], e[field])

    def test_preserves_exact_batches_and_full_factorial(self):
        for split in band.SPLITS:
            examples = [e for e in self.rows if e['split'] == split]
            ids = [e['example_id'] for e in examples]
            plan = dict(phases=[dict(name='primary', batches=[ids[i:i+8] for i in range(0, len(ids), 8)])])
            batches, references = band.select_batches(examples, plan)
            self.assertEqual(sum(map(len, batches)), 3456)
            self.assertTrue(all(b in batches for b in references))
            by_id = {e['example_id']: e for e in examples}
            fields = ('template', 'variant', 'role', 'alignment', 'query')
            self.assertEqual({tuple(by_id[eid][f] for f in fields) for b in batches for eid in b},
                             {tuple(by_id[eid][f] for f in fields) for b in references for eid in b})
            # A mixed selected/unselected batch must fail, not silently rebatch.
            broken = deepcopy(plan)
            selected_id = batches[0][0]
            other = next(e['example_id'] for e in examples if e['variant'] not in band.VARIANTS)
            a = next(b for b in broken['phases'][0]['batches'] if selected_id in b)
            b = next(b for b in broken['phases'][0]['batches'] if other in b)
            i, j = a.index(selected_id), b.index(other); a[i], b[j] = b[j], a[i]
            with self.assertRaises(AssertionError): band.select_batches(examples, broken)

    def test_joint_counts_keep_failed_controls_and_missing_pairs_fail(self):
        scores = task.score(self.rows, oracle(self.rows, recipient_messages=True))
        target = next(r for r in scores if r['variant'] == 'anchor_reminder' and r['query'] == 'private')
        target['correct'] = False
        pairs = audit.pairs(scores)
        primary = [r for r in pairs if r['contrast'] == 'anchor_original__anchor_reminder']
        self.assertEqual(len(primary), 216)
        self.assertEqual(sum(r['controls_both'] for r in primary), 215)
        with self.assertRaises(KeyError): audit.pairs([r for r in scores if r is not target])
        with self.assertRaises(AssertionError): audit.pairs(scores+[scores[0]])

    def test_qualification_requires_controls_both_splits_and_behavior(self):
        base = task.score(self.rows, oracle(self.rows))
        responses = oracle(self.rows)
        index = {e['example_id']: e for e in self.rows}
        for r in responses:
            e = index[r['example_id']]
            if e['variant'] == 'anchor_original' and e['query'] == 'message':
                r['response'] = e['recipient_optimal_answer']
        scores = task.score(self.rows, responses)
        self.assertTrue(band.qualification(scores, base)['passed'])
        broken = deepcopy(scores)
        next(r for r in broken if r['split'] == 'reserved' and r['variant'] == 'anchor_reminder' and r['query'] == 'fact0')['correct'] = False
        self.assertFalse(band.qualification(broken, base)['passed'])
        self.assertFalse(band.qualification(base, base)['passed'])
        with self.assertRaises(AssertionError): band.qualification(scores[:-8640], base)

    def test_actual_module_names_from_header(self):
        names = [f'base_model.model.model.language_model.layers.{i}.self_attn.{m}.lora_A.weight'
                 for i in (19, 20, 25, 31, 32) for m in ('q_proj', 'v_proj')]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); header = json.dumps({name: {} for name in names}).encode()
            (root/'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(header))+header)
            self.assertEqual(band.module_names(root), sorted(name.rsplit('.lora_A.weight', 1)[0]
                for name in names if any(f'.{i}.' in name for i in (20, 25, 31)) and '.v_proj.' in name))

    def test_adapter_provenance_allows_metadata_but_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                (root/name).write_text('original')
            expected = {name: band.digest(root/name) for name in
                        ('adapter_config.json', 'adapter_model.safetensors')}
            (root/'README.md').write_text('metadata')
            (root/'train_log.json').write_text(json.dumps({'seed': 1}))
            band.verify_adapter(root, expected, 1)
            with self.assertRaises(AssertionError): band.verify_adapter(root, expected, 0)
            (root/'adapter_model.safetensors').write_text('changed')
            with self.assertRaises(AssertionError): band.verify_adapter(root, expected, 1)


if __name__ == '__main__': unittest.main()
