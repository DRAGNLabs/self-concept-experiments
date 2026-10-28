"""Inspect every alpha-.75 private loss and all matched rotations, without new inference."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from analyze_explicit_payoff import audit_run,digest,read,write_json
from review_payoff_behavior import write_csv


def inspect(batch_path,out):
    b=json.loads(batch_path.read_text());assert b['status']=='completed_and_analyzed' and not b['errors']
    out.mkdir(parents=True,exist_ok=False);summary={};all_errors=[]
    for suite,paths in b['payoff_references'].items():
        n=1296 if suite=='validation' else 2592
        audit,data,inputs,scores,manifest=audit_run(Path(paths['offset_0.75']),out/suite/'audit',expected_n=n)
        comparisons={tag:{s['example_id']:s for s in read(Path(paths[tag])/'output/scored.jsonl')} for tag in ('base','adapter','offset_0.5','offset_1')}
        response={r['example_id']:r for r in read(Path(paths['offset_0.75'])/'output/responses.jsonl')}
        lookup={(e['instance_id'],e['role'],e['alignment'],e['query'],e['table_order'],e['answer_order']):(e,s) for e,s in zip(data,scores)}
        private=[];errors=[]
        for e,s in zip(data,scores):
            if e['query']!='private':continue
            table=e['instance']['table'];displayed=table[e['table_order']:]+table[:e['table_order']]
            rule_matches={}
            for name,func in (('total',sum),('minimum',min),('largest_single',max)):
                values=[func(r['payoffs']) for r in table];winners=[r['option'] for r,v in zip(table,values) if v==max(values)]
                rule_matches[name+'_match']=s['answer'] in winners
                rule_matches[name+'_unique']=len(winners)==1
            prefix=(e['instance_id'],e['role'],e['alignment'])
            rotations=[lookup[(*prefix,'private',t,a)][1] for t in range(3) for a in range(3)]
            opposite='aligned' if e['alignment']=='opposed' else 'opposed'
            other=lookup[(e['instance_id'],e['role'],opposite,'private',e['table_order'],e['answer_order'])][1]
            facts=[lookup[(*prefix,q,e['table_order'],e['answer_order'])][1] for q in ('fact0','fact1')]
            message=lookup[(*prefix,'message',e['table_order'],e['answer_order'])][1]
            r={'suite':suite,**s,**rule_matches,'chosen_table_position':next(i for i,row in enumerate(displayed) if row['option']==s['answer']),
               'chosen_answer_position':e['choices'].index(s['answer']),'assigned_table_position':next(i for i,row in enumerate(displayed) if row['option']==e['expected_answer']),
               'assigned_answer_position':e['choices'].index(e['expected_answer']),
               'rotation_correct':sum(x['correct'] for x in rotations),'rotation_n':9,
               'opposite_alignment_correct':other['correct'],'opposite_alignment_answer_changed':other['answer']!=s['answer'],
               'facts_both_correct':all(x['correct'] for x in facts),'message_answer':message['answer'],
               'message_same_as_private':message['answer']==s['answer'],
               **{tag+'_correct':xs[e['example_id']]['correct'] for tag,xs in comparisons.items()}}
            private.append(r)
            if not s['correct']:
                errors.append(r);all_errors.append({'suite':suite,'example':e,'score_and_context':r,'response':response[e['example_id']],
                    'all_rotations':rotations,'facts':facts,'message':message,'opposite_alignment':other})
        write_csv(out/suite/'all_private.csv',private);write_csv(out/suite/'private_losses.csv',errors)
        blocks=defaultdict(list)
        for r in private:blocks[r['instance_id'],r['role'],r['alignment']].append(r)
        blockrows=[{'instance_id':key[0],'role':key[1],'alignment':key[2],'n':len(rs),'correct':sum(r['correct'] for r in rs),
                    'distinct_answers':len({r['answer'] for r in rs})} for key,rs in blocks.items()]
        write_csv(out/suite/'rotation_blocks.csv',blockrows)
        summary[suite]={'n_private':len(private),'errors':len(errors),
            'by_role':dict(Counter(r['role'] for r in errors)),'by_alignment':dict(Counter(r['alignment'] for r in errors)),
            'by_instance':dict(Counter(r['instance_id'] for r in errors)),
            'chosen_table_positions':dict(Counter(r['chosen_table_position'] for r in errors)),
            'chosen_answer_positions':dict(Counter(r['chosen_answer_position'] for r in errors)),
            'assigned_table_positions':dict(Counter(r['assigned_table_position'] for r in errors)),
            'assigned_answer_positions':dict(Counter(r['assigned_answer_position'] for r in errors)),
            'error_rotation_blocks':[r for r in blockrows if r['correct']<r['n']],
            'error_counts':{key:sum(r[key] for r in errors) for key in ('other_client_optimal','recipient_optimal','dominated','facts_both_correct','opposite_alignment_correct','opposite_alignment_answer_changed','message_same_as_private','total_match','minimum_match','largest_single_match','base_correct','adapter_correct','offset_0.5_correct','offset_1_correct')}}
    assert sum(s['errors'] for s in summary.values())==18
    (out/'all_losses_full_context.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_errors))
    write_json(out/'summary.json',{'source_ledger':str(batch_path.resolve()),'source_ledger_sha256':digest(batch_path),'suites':summary})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();inspect(a.batch,a.out)
