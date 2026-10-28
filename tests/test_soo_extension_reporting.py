import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/soo/scripts'))
from extension_bridge_task import build, score
from extension_repeat_selection import TAGS
from extension_repeat_bridge import write, jsonl
from summarize_extension_repeat_bridge import bridge_summary, repeat_summary


class ReportingTests(unittest.TestCase):
    def test_opposing_role_effects_not_hidden_by_aggregate(self):
        examples=build('development'); scores=[]
        for tag in TAGS:
            responses=[dict(example_id=e['example_id'],completion_status='complete',response=(e['other_client_answer']
                if tag=='adapter' and e['query']=='recommend' and e['cue']==1 else e['assigned_goal_answer']
                if e['query'] in ('recommend','report') else e['expected_answer'])) for e in examples]
            scores += [dict(r,condition=tag) for r in score(examples,responses)]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); (p/'development').mkdir(); jsonl(p/'development/all_scores.jsonl',scores)
            rows=bridge_summary({'analysis':str(p)},'development',p)
            rows=[r for r in rows if r['condition']=='adapter' and r['metric']=='recipient_goal' and r['stock_alignment']=='all'
                  and r['contrast']=='speech_act_by_cue_interaction']
            self.assertEqual({r['recipient_alignment']:r['value'] for r in rows},{'all':0.0,'aligned':-1.0,'opposed':1.0})

    def test_repeat_semantics_separate_from_format_and_report_lost_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); (p/'data').mkdir()
            e=dict(example_id='history::x',source_group='history',query='history',focal_kind='target',honest_answer='kitchen',deceptive_answer='attic')
            jsonl(p/'data/repeat.jsonl',[e]); runs={}
            for t in TAGS:
                out=p/t; out.mkdir()
                before='attic' if t=='offset_0.6875' else 'kitchen'
                write(p/(t+'.json'),{'historical':{e['example_id']:dict(response=before,completion_status='complete')}})
                rows=[]
                for phase in ('fixed_1','fixed_2','shuffled_1','shuffled_2','singleton'):
                    after='the kitchen' if phase=='singleton' else before
                    rows.append(dict(example_id=e['example_id'],phase='history/'+phase,response=after,completion_status='complete'))
                jsonl(out/'planned_responses.jsonl',rows); runs[t]={'plan':t+'.json','output':str(out)}
            repeat_summary({'controller':str(p),'runs':{'repeat':runs}},p)
            rows=[json.loads(s) for s in (p/'paired_repeat_stability.jsonl').read_text().splitlines()]
            same=next(r for r in rows if r['reference']=='base' and r['condition']=='adapter')
            lost=next(r for r in rows if r['reference']=='adapter' and r['condition']=='offset_0.6875')
            self.assertTrue(same['both_original_answers_preserved_all_modes'])
            self.assertTrue(lost['originally_different']); self.assertFalse(lost['different_all_modes'])


if __name__=='__main__': unittest.main()
