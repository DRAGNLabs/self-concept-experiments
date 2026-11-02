import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/soo/scripts'))
import truth_wording_diagnostic as diag


class WordingDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {kind: diag.build(kind) for kind in diag.KINDS}
        cls.cross = {r['example_id']: r for r in cls.rows['cross']}

    def test_full_cross_independent_keys_and_spans(self):
        self.assertEqual(sum(map(len, self.rows.values())), 18720)
        ids = [r['example_id'] for rows in self.rows.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for kind, rows in self.rows.items():
            self.assertEqual(len({r['instance_id'] for r in rows}), 12)
            for e in rows:
                self.assertEqual(diag.independent_key(e), e['expected_answer'])
                self.assertEqual(e['table_order'], e['answer_order'])
                for spans in e['semantic_spans'].values():
                    for s in spans:
                        self.assertEqual(e['prompt'][s['start']:s['end']], s['text'])
        cross = self.rows['cross']
        for host, world, variant, role, alignment, relation in itertools.product(diag.NOUNS,
                range(12), diag.task.VARIANTS, (0, 1), ('aligned', 'opposed'), diag.task.RELATIONS):
            selected = [e for e in cross if e['template'] == host and e['instance_id'] == f'truthbenefit1-{world:02d}'
                        and e['variant'] == variant and e['role'] == role and e['alignment'] == alignment
                        and e['truth_relation'] == relation]
            self.assertEqual(len(selected), 7)
            self.assertEqual({(e['query'], e['table_order']) for e in selected},
                             {(q, 0) for q in diag.task.QUERIES} | {('fact_state', 1), ('fact_state', 2)})

    def test_swaps_only_change_one_section_and_keep_references(self):
        for kind in ('swap_t2', 'swap_t3'):
            for e in self.rows[kind]:
                original_id = f"truthdiag1-cross-{e['template']}-none-none::{e['parent_example_id']}"
                host = self.cross[original_id]
                a = {k: v[0]['text'] for k, v in e['semantic_spans'].items()}
                b = {k: v[0]['text'] for k, v in host['semantic_spans'].items()}
                self.assertEqual({k for k in a if a[k] != b[k]}, {e['component']})
                for noun in diag.NOUNS.values():
                    if noun != diag.NOUNS[e['template']]:
                        self.assertNotRegex(e['prompt'], r'\b'+noun+r'\b')
                for key in ('actual', 'choices', 'expected_answer', 'assigned_goal_answer', 'recipient_optimal_answer'):
                    self.assertEqual(e[key], host[key])

    def test_probes_keep_truth_and_do_not_edit_unrelated_sections(self):
        for e in self.rows['probes']:
            host = self.cross[f"truthdiag1-cross-{e['template']}-none-none::{e['parent_example_id']}"]
            a = {k: v[0]['text'] for k, v in e['semantic_spans'].items()}
            b = {k: v[0]['text'] for k, v in host['semantic_spans'].items()}
            if e['component'] == 'observation_only':
                self.assertEqual(a, {k: b[k] for k in ('state', 'question', 'format')})
            else:
                self.assertEqual({k for k in a if a[k] != b[k]}, {'question'})
                self.assertTrue(a['question'].startswith(b['question']))
            self.assertEqual(e['actual'], host['actual'])

    def test_complete_batch_sentinels_and_frozen_parent_anchors(self):
        parent = json.loads((ROOT/'experiments/soo/results/study2'/diag.PARENT).read_text())
        anchors, batches, raw, native = diag.anchors(parent)
        self.assertEqual(len(anchors), 128); self.assertEqual(len(batches), 16)
        for kind, rows in self.rows.items():
            together = rows + anchors
            plan = diag.make_plan(together, {e['example_id']: [1] for e in together}, batches)
            primary = plan['phases'][1]['batches']; sentinels = plan['phases'][0]['batches']
            self.assertEqual(primary[-16:], batches)
            self.assertTrue(all(len(b) == 8 for b in primary))
            self.assertEqual(sentinels, plan['phases'][2]['batches'])
            self.assertTrue(all(b in primary for b in sentinels))
            key = lambda e: tuple(e[k] for k in ('diagnostic_arm', 'template', 'component', 'donor_template', 'variant', 'role', 'query'))
            selected = {eid for b in sentinels for eid in b}
            self.assertEqual({key(e) for e in together}, {key(e) for e in together if e['example_id'] in selected})
            self.assertEqual(plan['generations'], len(together)+2*len(selected))

    def test_analysis_pairs_same_world_and_detects_gains_losses_and_tracking(self):
        scored = []
        for kind, rows in self.rows.items():
            responses = []
            for e in rows:
                # A deliberately mixed synthetic pattern: t0 truth, t2/t3 goal;
                # local_fact rescues them; reverse role swaps damage t0.
                answer = e['expected_answer']
                if e['query'] == 'fact_state' and e['template'] in ('t2', 't3'):
                    answer = e['assigned_goal_answer']
                if e['diagnostic_arm'] == 'probe' and e['component'] == 'local_fact': answer = e['actual']
                if e['diagnostic_arm'] == 'swap' and e['template'] == 't0' and e['component'] == 'role':
                    answer = e['assigned_goal_answer']
                responses.append(dict(example_id=e['example_id'], response=answer, completion_status='complete'))
            scored.extend(diag.score(rows, responses))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); result = diag.analyze(scored, out)
            self.assertEqual(result['diagnostic_responses'], 18720)
            self.assertTrue(result['no_adapter_release'])
            pairs = diag.read(out/'paired.jsonl')
            for r in pairs:
                if r['diagnostic_arm'] == 'probe' and r['component'] == 'local_fact' and r['template'] in ('t2', 't3'):
                    self.assertEqual(r['rescued'], r['truth_relation'] != 'assigned')
                if r['diagnostic_arm'] == 'swap' and r['template'] == 't0' and r['component'] == 'role':
                    self.assertEqual(r['damaged'], r['truth_relation'] != 'assigned')
            track = diag.read(out/'tracking.jsonl')
            self.assertTrue(all(r['all_three_correct'] for r in track if r['diagnostic_arm'] == 'probe' and r['component'] == 'local_fact'))
            self.assertTrue(all(r['all_three_assigned'] for r in track if r['diagnostic_arm'] == 'cross' and r['template'] in ('t2', 't3')))

    def test_failed_integrity_prevents_combined_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'batch.json'
            diag.engine.write(root/'anchor_reference.json', {})
            diag.engine.write(path, dict(status='diagnostic_submitted', controller=tmp, analysis=str(root/'analysis'),
                runs={k: {'base': {}} for k in diag.KINDS}, errors={}))
            with (patch.object(diag.engine, 'verify'), patch.object(diag.engine, 'audit_run', side_effect=AssertionError('sentinel')),
                  patch.object(diag, 'analyze') as analyze):
                diag.finish(path)
            analyze.assert_not_called()
            self.assertEqual(json.loads(path.read_text())['status'], 'execution_discrepancy_stopped')

    def test_rebatched_changes_are_reported_separately_from_integrity(self):
        rows = [e for e in self.rows['cross'] if e['template'] == e['world_template']]
        self.assertEqual(len(rows), 2016)
        old = [dict(example_id=e['parent_example_id'], response=e['expected_answer'],
                    completion_status='complete', generated_token_ids=[1], phase='primary') for e in rows]
        new = [dict(r, example_id=e['example_id']) for e, r in zip(rows, old)]
        new[0] = dict(new[0], response='changed', generated_token_ids=[2])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); parent = out/'parent.json'
            diag.engine.write(parent, dict(runs={'base-'+s: {'output': s} for s in diag.parent_flow.SPLITS}))
            batch = dict(parent=str(parent), runs={'cross': {'base': {'output': 'new'}}})
            def responses(path):
                if path.parent.name == 'new': return new
                return [r for e, r in zip(rows, old) if e['split'] == path.parent.name]
            with patch.object(diag, 'read', side_effect=responses):
                result = diag.compare_rebatched_parent(batch, rows, out)
            self.assertEqual(result, dict(n=2016, answer_changed=1, tokens_changed=1))


if __name__ == '__main__': unittest.main()
