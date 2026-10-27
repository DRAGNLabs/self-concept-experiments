"""Checks for fixed-dose transfer, paired losses and historical identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from payoff_half_transfer import analyze,paired_counts,selected_dose_checks
from launch_goals_roles import freeze


class HalfTransferTests(unittest.TestCase):
    def test_equal_aggregate_accuracy_does_not_hide_paired_disagreements(self):
        a=[{'example_id':'a','answer':'x','correct':True},{'example_id':'b','answer':'x','correct':False}]
        b=[{'example_id':'a','answer':'y','correct':False},{'example_id':'b','answer':'y','correct':True}]
        r=paired_counts(a,b)
        self.assertEqual(r,{'n':2,'answer_differences':2,'first_correct_second_wrong':1,'first_wrong_second_correct':1})
        with self.assertRaises(AssertionError):paired_counts(a,b[::-1])

    def test_half_dose_checks_expect_selected_history_not_full_strength_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new'
            records=[{'scenario':'main','label':'deceptive','generated_token_ids':[i,1]} for i in range(32)]
            for p in (old,new):
                (p/'output').mkdir(parents=True);(p/'band_deltas.pt').write_bytes(b'unchanged offsets')
                (p/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            manifest={'source_sha256':{'source':'abc'},'condition':'offset','offset_alpha':.5}
            (old/'output/manifest.json').write_text(json.dumps(manifest))
            result=({'repeat_passed':True,'regressions_passed':False},[],[],[],manifest)
            self.assertTrue(selected_dose_checks(result,new,old)['passed'])
            records[0]['generated_token_ids']=[999,1]
            (new/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertFalse(selected_dose_checks(result,new,old)['passed'])
            (new/'band_deltas.pt').write_bytes(b'changed offsets')
            with self.assertRaises(AssertionError):selected_dose_checks(result,new,old)

    def test_old_protocol_and_other_doses_cannot_launch_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for dataset in ('payoff_preferences1','explicit_payoff1_validation'):
                (p/'manifest.json').write_text(json.dumps({'dataset_id':dataset,'task':'explicit_payoff','sha256':{}}))
                for protocol,alpha in ((None,.5),('payoff_preferences1',.5),('payoff_half_transfer1',.75)):
                    with self.subTest(dataset=dataset,protocol=protocol,alpha=alpha),self.assertRaises(ValueError):
                        freeze('offset',data_dir=p,offset_alpha=alpha,followup_protocol=protocol)

    def test_other_suite_is_analyzed_even_if_one_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original/data/goals_roles_pilot';original.mkdir(parents=True)
            for suite in ('main','checks'):(original/(suite+'.jsonl')).write_text('')
            batch_path=root/'batch.json'
            b={'status':'submitted','controller':str(root),'controller_sha256':{},'source_ledgers_sha256':{},
               'original_development':str(root/'original'),'selected_dose':'unused','analysis':str(root/'analysis'),
               'suites':{name:{'half':{'status':'submitted'}} for name in ('validation','numerical')}}
            batch_path.write_text(json.dumps(b))
            with patch('payoff_half_transfer.analyze_suite',side_effect=[ValueError('missing output'),{'checks':{'passed':True}}]) as run:
                with self.assertRaises(RuntimeError):analyze(batch_path)
                self.assertEqual(run.call_count,2)
            final=json.loads(batch_path.read_text())
            self.assertEqual(set(final['results']),{'numerical'})
            self.assertIn('validation',final['errors'])
            self.assertEqual(final['status'],'execution_error_requires_inspection')


if __name__=='__main__':unittest.main()
