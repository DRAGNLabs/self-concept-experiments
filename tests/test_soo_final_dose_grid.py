"""Final-grid dose boundaries and cross-task execution identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/soo/scripts'))
from final_dose_grid import DOSES,NEW_TAGS,verify_condition,same_dose_history,payoff_analysis
from analyze_explicit_payoff import digest
from launch_goals_roles import freeze
from unittest.mock import patch


class FinalGridTests(unittest.TestCase):
    def test_only_final_grid_allowed_no_protocol_expansion(self):
        self.assertEqual(DOSES,(.5625,.625,.6875))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for dataset in ('explicit_payoff1_development','explicit_payoff1_validation','payoff_preferences1'):
                (p/'manifest.json').write_text(json.dumps({'dataset_id':dataset,'task':'explicit_payoff','sha256':{}}))
                for protocol,condition,alpha in [('final_dose_grid1','offset',.6),('final_dose_grid1','offset',.5),('final_dose_grid1','adapter',.625),('cross_task_dose1','offset',.625),(None,'offset',.625)]:
                    with self.subTest(dataset=dataset,protocol=protocol,alpha=alpha),self.assertRaises(ValueError):
                        freeze(condition,data_dir=p,offset_alpha=alpha,followup_protocol=protocol)

    def test_wrong_dose_artifact_and_repeat_are_execution_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'band_deltas.pt').write_bytes(b'offset')
            b={'offset_sha256':digest(p/'band_deltas.pt'),'inference_source_sha256':{'source':'hash'}}
            m={'condition':'offset','offset_alpha':.625,'source_sha256':b['inference_source_sha256']}
            result=({'repeat_passed':True},[],[],[],m)
            verify_condition(result,p,'offset_0.625',b)
            with self.assertRaises(AssertionError):verify_condition(result,p,'offset_0.5625',b)
            with self.assertRaises(AssertionError):verify_condition(({'repeat_passed':False},[],[],[],m),p,'offset_0.625',b)
            (p/'band_deltas.pt').write_bytes(b'changed')
            with self.assertRaises(AssertionError):verify_condition(result,p,'offset_0.625',b)

    def test_same_dose_history_requires_tokens_not_label_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            records=[{'example_id':str(i),'label':'deceptive','generated_token_ids':[i,1]} for i in range(32)]
            for path in (a,b):
                (path/'output').mkdir(parents=True)
                (path/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            same_dose_history(a,b)
            records[0]['generated_token_ids']=[99,1]
            (a/'output/regressions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(AssertionError):same_dose_history(a,b)

    def test_one_failed_run_does_not_suppress_other_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('final_dose_grid.audit_run',side_effect=RuntimeError('missing output')) as audit:
                result,runs=payoff_analysis('validation',{tag:tag for tag in NEW_TAGS},{},Path(tmp)/'analysis')
                self.assertEqual(audit.call_count,3);self.assertEqual(set(result['errors']),set(NEW_TAGS))
                self.assertFalse(result['checks_passed']);self.assertEqual(runs,{})


if __name__=='__main__':unittest.main()
