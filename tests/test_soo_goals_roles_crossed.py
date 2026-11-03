"""Validate the new crossings against unchanged v2 prompts and causal task rules."""

from collections import Counter
import unittest

from selfconcept.soo.goals_roles import build_dataset as original_dataset
from selfconcept.soo.goals_roles_crossed import build_dataset


class CrossedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_dataset()
        cls.lookup = {(r['base_world_id'], r['world']['wording'], r['world']['belief_swapped'],
                       r['world']['target0_option'], r['role'], r['alignment'], r['query'], r['orientation']): r
                      for r in cls.rows}

    def test_full_crossing_and_original_cells(self):
        self.assertEqual(len(self.rows), 5376)
        self.assertEqual(len(self.lookup), len(self.rows))
        self.assertEqual(len({r['example_id'] for r in self.rows}), len(self.rows))
        self.assertEqual(set(Counter(r['base_world_id'] for r in self.rows).values()), {448})
        original = {r['example_id']: r for r in original_dataset(2)}
        retained = [r for r in self.rows if r['original_factor_cell']]
        self.assertEqual(len(retained), 672)
        for r in retained:
            old = original[r['reference_v2_example_id']]
            for field in ('prompt', 'expected_answer', 'outcomes', 'semantic_spans', 'choices'):
                self.assertEqual(r[field], old[field])

    def test_factor_interventions_change_only_intended_semantics(self):
        for k, a in self.lookup.items():
            w, wording, belief, layout, role, alignment, query, orientation = k
            paraphrase = self.lookup[(w, 1-wording, belief, layout, role, alignment, query, orientation)]
            self.assertEqual(a['expected_answer'], paraphrase['expected_answer'])
            for span in a['semantic_spans']:
                if span != 'role':
                    self.assertEqual([s['text'] for s in a['semantic_spans'][span]],
                                     [s['text'] for s in paraphrase['semantic_spans'][span]])
            counterbelief = self.lookup[(w, wording, not belief, layout, role, alignment, query, orientation)]
            self.assertEqual(a['expected_answer'] == counterbelief['expected_answer'],
                             query not in ('prediction', 'belief_check'))
            counterlayout = self.lookup[(w, wording, belief, 1-layout, role, alignment, query, orientation)]
            self.assertEqual(a['expected_answer'] == counterlayout['expected_answer'],
                             query in ('role_check', 'goal_check'))

    def test_payoffs_and_answer_balance_within_each_factor_cell(self):
        counts = Counter()
        for r in self.rows:
            w = r['world']
            if r['query'] in ('message', 'private'):
                desired_room = w['target0_option'] if r['role'] == 0 else 1-w['target0_option']
                selected = 1-desired_room if r['family'] == 'burglar' else desired_room
                self.assertEqual(r['expected_answer'], w['options'][selected])
            counts[(r['family'], w['wording'], w['belief_swapped'], w['target0_option'],
                    r['query'], r['choices'].index(r['expected_answer']))] += 1
            for spans in r['semantic_spans'].values():
                for span in spans:
                    self.assertEqual(r['prompt'][span['start']:span['end']], span['text'])
        for k, value in counts.items():
            self.assertEqual(value, 24)
            self.assertEqual(value, counts[(*k[:-1], 1-k[-1])])


if __name__ == '__main__':
    unittest.main()
