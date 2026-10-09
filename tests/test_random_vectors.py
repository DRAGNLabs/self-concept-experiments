"""Random controls: pooling equivalence, sign symmetry, matching and clustered statistics."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from selfconcept.measurement.capture import capture
from selfconcept.correlation.random_vectors import random_directions, pooled_scorer
from selfconcept.correlation.random_vector_analysis import analyze_cell, adjusted_correlations, baseline_aurocs, pair_totals, compare_to_random, load_cache, project, main as analyze_main
from selfconcept.correlation.rh_hillclimb import LabeledRecord, WorklistSettings, strict_transcripts


class RandomControlTests(unittest.TestCase):
    def test_unit_vectors_reproducible_prefix_and_independent_layers(self):
        directions = random_directions(64, 32)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(directions, random_directions(64, 64)[:32])
        self.assertFalse(np.array_equal(directions, random_directions(64, 32, layer=18)))

    def test_pooling_matches_token_projection_at_real_decoder_hook(self):
        torch.set_num_threads(1)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
        pool, counts = pooled_scorer({'prompt': [0, 1], 'cot': [2], 'final': [3, 4, 5]}, 6)
        with torch.inference_mode(), capture(model, [1]) as full, capture(model, [1], pool) as pooled:
            model(torch.tensor([[1, 2, 3, 4, 5, 6]]))
        directions = torch.tensor(random_directions(16, 8)).T
        for i, indices in enumerate(([0, 1], [2], [3, 4])):
            torch.testing.assert_close(pooled[1][0][i] @ directions, (full[1][0][indices] @ directions).mean(0))
        self.assertEqual(counts, [2, 1, 2])
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_pair_weighting_ties_and_problem_cluster(self):
        scores = np.array([[1, -1], [0, 0], [1, -1], [1, -1], [0, 0], [2, -2]])
        labels = np.array([1, 0, 1, 0, 0, 1])
        strata = ['a', 'a', 'b', 'b', 'b', 'c']
        problems = ['p', 'p', 'p', 'p', 'p', 'q']
        u, pairs, n = pair_totals(scores, labels, strata, problems)
        self.assertEqual(n, 2)
        self.assertEqual(len(pairs), 1)
        np.testing.assert_allclose(u.sum(0) / pairs.sum(), [2.5/3, .5/3])
        result = analyze_cell(scores, labels, np.zeros((6, 3)), strata, problems, bootstrap=10)
        self.assertEqual(result['n_problems'], 1)
        self.assertAlmostEqual(result['assistant_axis_problem_bootstrap_ci'][0], 2.5/3)

    def test_identical_residuals_tie_exactly(self):
        residuals = np.tile(np.random.default_rng(0).normal(size=(1, 64)).astype(np.float32) * 5000, (40, 1))
        scores = project(residuals, random_directions(64, 16))
        self.assertTrue((scores == scores[0]).all())
        u, pairs, _ = pair_totals(scores, np.arange(40) % 2, ['a'] * 40, ['p'] * 40)
        np.testing.assert_array_equal(u.sum(0) / pairs.sum(), .5)

    def test_per_direction_intervals_and_predictivity(self):
        # Four problems with unequal pair counts; each has two repeated strata.
        # The columns are AA, a positive predictor, its inverse, ties, and a
        # direction whose high point estimate remains uncertain across problems.
        scores, labels, strata, problems = [], [], [], []
        for p, negatives in enumerate([1, 2, 3, 4]):
            for turn in range(2):
                for label in [1, *([0] * negatives)]:
                    scores.append([label if p else -label, label, -label, 0, label if p else -label])
                    labels.append(label)
                    strata.append(f'{p}:{turn}')
                    problems.append(str(p))
        scores, labels = np.asarray(scores), np.asarray(labels)
        result = analyze_cell(scores, labels, np.zeros((len(labels), 3)), strata, problems,
                              bootstrap=2000, seed=42)
        cis = np.asarray(result['all_auroc_problem_bootstrap_cis'])
        np.testing.assert_array_equal(cis[1:4], [[1, 1], [0, 0], [.5, .5]])
        self.assertGreater(result['all_aurocs'][4], .8)
        self.assertLessEqual(cis[4, 0], .5)
        self.assertGreater(cis[4, 1], .5)
        self.assertEqual(result['random_direction_predictivity'], {
            'n_random': 4, 'n_above_chance': 1, 'n_below_chance': 1,
            'n_excluding_chance': 2, 'fraction_excluding_chance': .5})
        # Independently resample whole problems, retaining all their strata and
        # weighting by pair count. The old AA-only CI must remain identical.
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(2000):
            selected = rng.integers(4, size=4)
            weights = (selected + 1) * 2
            draws.append(np.sum(weights * (selected != 0)) / weights.sum())
        np.testing.assert_allclose(cis[4], np.quantile(draws, [.025, .975]))
        np.testing.assert_allclose(result['assistant_axis_problem_bootstrap_ci'], np.quantile(draws, [.025, .975]))
        self.assertEqual(result['assistant_axis_problem_bootstrap_ci'], cis[0].tolist())
        flipped = analyze_cell(-scores, labels, np.zeros((len(labels), 3)), strata, problems,
                               bootstrap=2000, seed=42)
        np.testing.assert_allclose(flipped['all_auroc_problem_bootstrap_cis'], 1 - cis[:, ::-1])
        self.assertEqual(flipped['random_direction_predictivity']['n_excluding_chance'], 2)

    def test_disabled_bootstrap_and_no_matched_pairs(self):
        scores = np.array([[1, 1], [0, 0]])
        result = analyze_cell(scores, np.array([1, 0]), np.zeros((2, 3)), ['a', 'a'], ['p', 'p'], bootstrap=0)
        self.assertIsNone(result['assistant_axis_problem_bootstrap_ci'])
        self.assertIsNone(result['all_auroc_problem_bootstrap_cis'])
        self.assertIsNone(result['random_direction_predictivity'])
        self.assertEqual(result['all_aurocs'], [1, 1])
        unmatched = analyze_cell(scores, np.array([1, 0]), np.zeros((2, 3)), ['a', 'b'], ['p', 'p'])
        self.assertEqual(unmatched['status'], 'no_matched_pairs')

    def test_baselines_score_token_count_and_residual_norm(self):
        residuals = np.array([[3, 4], [0, 1], [6, 8], [1, 0]], dtype=np.float32)
        result = baseline_aurocs(np.array([1, 2, 3, 4]), residuals, np.array([1, 0, 1, 0]), ['a', 'a', 'b', 'b'], ['p', 'p', 'q', 'q'])
        self.assertEqual(result, {'region_tokens': 0., 'residual_norm': 1.})

    def test_random_comparison_is_sign_invariant(self):
        x = np.array([.9, .1, .8, .5])
        self.assertEqual(compare_to_random(x, .5)['n_random_at_least_as_extreme'], 1)
        self.assertEqual(compare_to_random(x, .5), {**compare_to_random(1-x, .5),
                          'assistant_axis': .9, 'random_quantiles': compare_to_random(x, .5)['random_quantiles']})

    def test_length_confound_is_removed(self):
        lengths = np.column_stack([np.arange(40), np.zeros(40), np.zeros(40)])
        scores = np.column_stack([np.arange(40), np.arange(40)[::-1]])
        labels = (np.arange(40) > 19).astype(int)
        result = adjusted_correlations(scores, labels, lengths, ['a']*40)
        self.assertTrue(np.isnan(result).all())

    def test_strict_matching_excludes_family_and_short_episode(self):
        def item(run, label, turn, last):
            record = {'example_id': 'p', 'scenario': 'evilgenie', 'max_attempts': 5,
                      'attempt_budget_stated': True, 'feedback': 'neutral',
                      'attempts': [{'truncated': False} for _ in range(last+1)]}
            return LabeledRecord(run, run, 'tag_evilgenie', record, [], label, turn)
        records = [item('n_a_seed0', 1, 2, 3), item('n_a_seed1', 1, 3, 3),
                   item('n_a_seed2', 0, 4, 4), item('n_a_seed3', 0, 0, 0), item('n_b_seed0', 0, 4, 4)]
        def build(root, selected):
            return [{'example_id': r.run, 'scenario': 'evilgenie', 'turn': r.turn, 'outcome': {}} for r in selected]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'data').mkdir()
            (root/'data/tag.jsonl').write_text(json.dumps({'example_id': 'p'})+'\n')
            with patch('selfconcept.correlation.rh_hillclimb.load_worklists', return_value={(r.run, 'tag'): WorklistSettings({}, None) for r in records}), \
                 patch('selfconcept.correlation.rh_hillclimb.conversation_before_turn', return_value=[{'role': 'user', 'content': 'same'}]), \
                 patch('selfconcept.correlation.rh_hillclimb.build_transcripts', side_effect=build):
                rows = strict_transcripts(root, records)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['outcome']['stratum'] for r in rows}), 2)
        self.assertEqual(len({r['example_id'] for r in rows}), 4)
        self.assertEqual(sum(r['outcome']['episode_id'] == 'n_a_seed2' for r in rows), 2)

    def test_analysis_end_to_end_and_missing_shard_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root/'shard_0'
            shard.mkdir()
            identity = {'model': 'tiny', 'shards': 1, 'shard': 0, 'layers': [1], 'limit': None, 'n_expected': 4}
            manifest = {'identity': identity, 'stage': 'complete', 'num_layers': 2, 'hidden_size': 4}
            (shard/'manifest.json').write_text(json.dumps(manifest))
            rows = []
            for i in range(4):
                means = np.zeros((3, 4), dtype=np.float32)
                means[:, 0] = i % 2
                means[:, 1:] = np.random.default_rng(i).normal(size=(3, 3))
                np.savez(shard/f'{i}.npz', layer_1=means)
                rows.append({'example_id': str(i), 'scenario': 'test', 'status': 'complete',
                             'residuals': f'{i}.npz', 'region_counts': [2, 2, 2], 'prompt_tokens': 10,
                             'outcome': {'status': 'complete', 'scenario': 'test', 'label': 'x',
                                         'verdict': 'HACK' if i % 2 else 'unflagged',
                                         'problem': str(i//2), 'stratum': str(i//2)}})
            (shard/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            axis = torch.zeros(2, 4)
            axis[:, 0] = 1
            torch.save(axis, root/'axis.pt')
            argv = ['analyze', str(shard), '--axis', str(root/'axis.pt'), '--out', str(root/'analysis'),
                    '--random-count', '8', '--bootstrap', '10']
            with patch.object(sys, 'argv', argv):
                analyze_main()
            result = json.loads((root/'analysis/analysis.json').read_text())
            primary = next(c for c in result['cells'] if c['primary'])
            self.assertEqual(primary['auroc']['assistant_axis'], 1)
            self.assertEqual(primary['baseline_aurocs']['region_tokens'], .5)
            self.assertEqual(result['bootstrap'], 10)
            self.assertEqual(len(primary['all_auroc_problem_bootstrap_cis']), 9)
            self.assertEqual(primary['random_direction_predictivity']['n_random'], 8)
            self.assertIn('Random CI excludes 0.5', (root/'analysis/analysis.md').read_text())
            self.assertTrue((root/'analysis/random_controls.png').exists())
            identity['shards'] = 2
            (shard/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'every shard'):
                load_cache([shard])


if __name__ == '__main__':
    unittest.main()
