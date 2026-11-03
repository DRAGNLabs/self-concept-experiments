"""Fixed-dose presentation guards and execution-discrepancy handling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from analyze_explicit_payoff import digest
from launch_goals_roles import freeze
from payoff_calibrated_presentation import DOSES, PROTOCOL, SETTING_FIELDS, audit_conditions, fixed_dose_checks, match_presentation_inputs


class CalibratedPresentationTests(unittest.TestCase):
    def test_input_matching_rejects_unrelated_source_and_batch_changes(self):
        fields = dict.fromkeys(SETTING_FIELDS, 'identical')
        old_source = {'soo/run_goals_roles.py':'old', 'soo/payoff_diagnostic.py':'same', 'generation.py':'same'}
        new_source = dict(old_source, **{'soo/run_goals_roles.py':'new', 'soo/payoff_preferences.py':'added'})
        old = (None, ['data'], ['tokens'], None, dict(fields, source_sha256=old_source))
        new = (None, ['data'], ['tokens'], None, dict(fields, source_sha256=new_source))
        match_presentation_inputs(new, old)
        with self.assertRaises(AssertionError):
            match_presentation_inputs(new, (*old[:4], dict(old[4], batches='changed')))
        with self.assertRaises(AssertionError):
            match_presentation_inputs(new, (*old[:4], dict(old[4], source_sha256=dict(old_source, **{'generation.py':'changed'}))))

    def test_scope_rejects_retuning_and_other_tasks(self):
        self.assertEqual(DOSES, (0.5, 0.6875))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for dataset in ('payoff_name_column1', 'explicit_payoff1_development'):
                (path/'manifest.json').write_text(json.dumps(dict(dataset_id=dataset,
                    diagnostic_id='payoff_name_column1' if dataset=='payoff_name_column1' else None,
                    task='explicit_payoff', sha256={})))
                cases = [('offset', .625), ('offset', .75), ('adapter', .6875), ('base', .5)]
                if dataset != 'payoff_name_column1': cases.append(('offset', .6875))
                for condition, alpha in cases:
                    with self.subTest(dataset=dataset, condition=condition, alpha=alpha), self.assertRaises(ValueError):
                        freeze(condition, data_dir=path, offset_alpha=alpha, followup_protocol=PROTOCOL)

    def test_same_dose_checks_use_exact_records_not_full_dose_honesty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, previous = Path(tmp)/'new', Path(tmp)/'previous'
            source = {'runner.py': 'unchanged'}
            manifest = dict(condition='offset', offset_alpha=.5, source_sha256=source)
            records = [{'example_id':str(i), 'label':'deceptive', 'generated_token_ids':[i, 1]} for i in range(32)]
            for run in (path, previous):
                (run/'output').mkdir(parents=True)
                (run/'output/manifest.json').write_text(json.dumps(manifest))
                (run/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                (run/'band_deltas.pt').write_bytes(b'fixed offset')
            summary = dict(sentinel_repeat_passed=True, historical_n=432,
                           historical_answer_changes=0, historical_token_changes=0, regressions_passed=False)
            offset_hash = digest(path/'band_deltas.pt')
            self.assertTrue(fixed_dose_checks(path, previous, .5, summary, source, offset_hash)['passed'])
            for change in (dict(historical_token_changes=1), dict(historical_answer_changes=1),
                           dict(sentinel_repeat_passed=False), dict(historical_n=431)):
                with self.subTest(change=change), self.assertRaises(AssertionError):
                    fixed_dose_checks(path, previous, .5, dict(summary, **change), source, offset_hash)
            with self.assertRaises(AssertionError):
                fixed_dose_checks(path, previous, .6875, summary, source, offset_hash)
            records[0]['generated_token_ids'] = [99]
            (path/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(AssertionError):
                fixed_dose_checks(path, previous, .5, summary, source, offset_hash)

    def test_one_missing_output_does_not_suppress_other_audits(self):
        entries = {str(a):dict(path='missing', historical='old', condition='offset', alpha=a) for a in DOSES}
        with patch('payoff_calibrated_presentation.diagnostic_analysis', side_effect=RuntimeError('missing output')) as audit:
            results, runs, errors = audit_conditions(entries, {}, Path('unused'))
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(set(errors), set(entries))
        self.assertFalse(results)
        self.assertFalse(runs)


if __name__ == '__main__': unittest.main()
