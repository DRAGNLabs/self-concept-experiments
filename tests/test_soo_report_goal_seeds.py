import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/soo/scripts'))
import report_goal_seeds as workflow


class SeedReplicationTests(unittest.TestCase):
    def test_intended_adapter_not_merely_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter = root / 'adapter'; adapter.mkdir()
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'seed-one-weights')
            (adapter / 'train_log.json').write_text('{"seed": 1}')
            hashes = {name: workflow.digest(adapter / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}
            batch = dict(controller=tmp, seed=1, adapter_sha256=hashes)
            manifest = dict(adapter=str(adapter), artifact_sha256={str(adapter / k): v for k, v in hashes.items()})
            workflow.adapter_binding(batch, manifest)
            wrong = copy.deepcopy(manifest); wrong['adapter'] = str(root / 'seed0')
            with self.assertRaises(AssertionError): workflow.adapter_binding(batch, wrong)
            (adapter / 'adapter_model.safetensors').write_bytes(b'seed-two-weights')
            wrong = copy.deepcopy(manifest)
            wrong['artifact_sha256'][str(adapter / 'adapter_model.safetensors')] = workflow.digest(adapter / 'adapter_model.safetensors')
            with self.assertRaises(AssertionError): workflow.adapter_binding(batch, wrong)

    def test_new_seed_historical_changes_are_findings_not_failures(self):
        reference = [dict(example_id=str(i), orientation='original', scenario='main',
            prompt_tokens=100, generated_tokens=1, generated_token_ids=[1],
            completion_status='complete', response='A', label='honest') for i in range(32)]
        current = copy.deepcopy(reference)
        current[0].update(generated_token_ids=[2], response='B', label='deceptive')
        findings = workflow.regression_comparisons(current, reference)
        self.assertEqual(sum(r['answer_changed'] for r in findings), 1)
        self.assertEqual(sum(r['tokens_changed'] for r in findings), 1)
        with self.assertRaises(AssertionError): workflow.regression_comparisons(current[:-1], reference)
        bad = copy.deepcopy(current); bad[-1] = bad[0]
        with self.assertRaises(AssertionError): workflow.regression_comparisons(bad, reference)
        bad = copy.deepcopy(current); bad[0]['prompt_tokens'] += 1
        with self.assertRaises(AssertionError): workflow.regression_comparisons(bad, reference)

    def test_null_and_reversed_seed_effects_are_retained(self):
        def summary(original, reminder):
            return [dict(condition=tag, variant=v, query='message', alignment='opposed', n=54, recipient_goal=n)
                for tag, ns in [('base', (10, 12)), ('adapter', (original, reminder))]
                for v, n in zip(('original', 'original_reminder'), ns)]
        parent = dict(results={s: dict(summary=summary(54, 0)) for s in workflow.SPLITS})
        results = {str(seed): {s: dict(summary=summary(*counts)) for s in workflow.SPLITS}
                   for seed, counts in [(1, (20, 20)), (2, (0, 54))]}
        result = workflow.summarize_all(parent, results)
        self.assertEqual(len(result['primary_counts']), 12)
        self.assertEqual(len(result['primary_effects']), 6)
        by_seed = {r['seed']: r for r in result['primary_effects'] if r['split'] == 'development'}
        self.assertEqual(by_seed[0]['adapter_change'], -1)
        self.assertEqual(by_seed[1]['adapter_change'], 0)
        self.assertEqual(by_seed[2]['adapter_change'], 1)
        self.assertAlmostEqual(by_seed[1]['baseline_adjusted_change'], -2/54)

    def test_integrity_failure_preserves_other_audits_and_stops_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root / 'parent.json'; parent.write_text('{}')
            children = {}
            for seed in workflow.SEEDS:
                p = root / f'{seed}.json'; p.write_text(json.dumps({'seed': seed})); children[str(seed)] = str(p)
            batch = dict(status='submitted', parent=str(parent), analysis=str(root / 'analysis'),
                children=children, runs={f'seed{s}-{k}': {} for s in workflow.SEEDS for k in workflow.SPLITS})
            p = root / 'batch.json'; workflow.write(p, batch)
            def audit(child, split, parent):
                if child['seed'] == 1 and split == 'development': raise AssertionError('sentinel changed')
                return dict(historical=[], checks_passed=True)
            with patch.object(workflow.engine, 'verify'), patch.object(workflow, 'parent_audits'), \
                 patch.object(workflow, 'audit_seed', side_effect=audit) as check, \
                 patch.object(workflow.parent_workflow, 'analyze') as analyze:
                workflow.advance(p)
            self.assertEqual(check.call_count, 4)
            analyze.assert_not_called()
            result = json.loads(p.read_text())
            self.assertEqual(result['status'], 'execution_discrepancy_stopped')
            self.assertEqual(result['runs']['seed2-reserved']['status'], 'completed_and_audited')
            self.assertIn('seed1-development', result['errors'])


if __name__ == '__main__': unittest.main()
