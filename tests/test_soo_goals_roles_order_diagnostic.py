import unittest
from collections import Counter

from selfconcept.soo.goals_roles_crossed import build_dataset as crossed
from selfconcept.soo.goals_roles_order_diagnostic import build_dataset


class OrderDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_dataset()

    def test_original_orders_exactly_reproduce_crossed_prompts(self):
        originals = {r['example_id']: r for r in crossed()}
        tied = [r for r in self.rows if r['reference_crossed_example_id']]
        self.assertEqual(len(self.rows), 3072)
        self.assertEqual(len(tied), 768)
        self.assertEqual(len({r['example_id'] for r in self.rows}), 3072)
        for r in tied:
            old = originals[r['reference_crossed_example_id']]
            for f in ('prompt', 'semantic_spans', 'choices', 'expected_answer', 'outcomes'):
                self.assertEqual(r[f], old[f])

    def test_each_order_toggle_changes_only_its_own_clause(self):
        groups = {}
        for r in self.rows:
            k = (r['world_id'], r['role'], r['alignment'], r['query'])
            groups.setdefault(k, []).append(r)
        mapping = {'facts': 'facts', 'beliefs': 'beliefs', 'answers': 'format'}
        self.assertEqual(len(groups), 384)
        for group in groups.values():
            self.assertEqual(len(group), 8)
            for a in group:
                for factor, kind in mapping.items():
                    wanted = {**a['order_factors'], factor: 1-a['order_factors'][factor]}
                    b = next(r for r in group if r['order_factors'] == wanted)
                    self.assertEqual(a['expected_answer'], b['expected_answer'])
                    self.assertEqual(a['outcomes'], b['outcomes'])
                    for span in a['semantic_spans']:
                        aa = [s['text'] for s in a['semantic_spans'][span]]
                        bb = [s['text'] for s in b['semantic_spans'][span]]
                        self.assertEqual(aa == bb, span != kind)
                    self.assertEqual(a['choices'] == b['choices'], factor != 'answers')

    def test_balanced_keys_and_exact_spans(self):
        positions = Counter()
        for r in self.rows:
            positions[(r['query'], tuple(r['order_factors'].values()), r['choices'].index(r['expected_answer']))] += 1
            for spans in r['semantic_spans'].values():
                for s in spans:
                    self.assertEqual(r['prompt'][s['start']:s['end']], s['text'])
        self.assertEqual(set(positions.values()), {48})


if __name__ == '__main__':
    unittest.main()
