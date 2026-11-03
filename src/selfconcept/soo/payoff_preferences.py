"""Six feasible combinations of total, minimum and largest payoff optima.

Derived from frozen development prompts; changes only displayed payoff numbers.
Every client retains its original unique optimum and the third option is dominated.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .explicit_payoff import keys
from .goals_roles import PromptBuilder

# Payoffs for client-0's optimum and client-1's optimum, respectively.
# The sum winner must agree with at least one of the minimum/maximum winners;
# the two other combinations are mathematically impossible for two options.
PROFILES = (((12,5),(2,10)), ((10,6),(1,12)), ((14,1),(4,10)),
            ((10,4),(1,14)), ((12,1),(6,10)), ((10,2),(5,12)))
RULES = {'total':sum, 'minimum':min, 'largest_single':max}


def rule_keys(instance):
    result={}
    for name, function in RULES.items():
        values={r['option']:function(r['payoffs']) for r in instance['table']}
        best=max(values.values())
        winners=[label for label,value in values.items() if value==best]
        assert len(winners)==1
        result[name]=winners[0]
    return result


def transform(source,profile):
    row=deepcopy(source)
    instance=row['instance']
    original=keys(instance,0,'aligned','private')
    scale=1+int(source['instance_id'].rsplit('-',1)[1])
    assert scale in (1,2,3)
    for option in instance['table']:
        if option['option']==original['assigned_goal_answer']:
            option['payoffs']=[scale*v for v in PROFILES[profile][0]]
        elif option['option']==original['other_role_answer']:
            option['payoffs']=[scale*v for v in PROFILES[profile][1]]
        else:option['payoffs']=[0,0]
    instance['instance_id']=f"pref1-p{profile}-"+source['instance_id']
    p=PromptBuilder()
    clauses=sorted((s['start'],kind,s['text']) for kind,spans in source['semantic_spans'].items() for s in spans)
    for _,kind,text in clauses:
        if kind=='payoffs':
            table=instance['table'][source['table_order']:]+instance['table'][:source['table_order']]
            text='The payoff table lists the points paid to each client:\nOption | '+' | '.join(instance['clients'])
            text+='\n'+'\n'.join(r['option']+' | '+' | '.join(map(str,r['payoffs'])) for r in table)
        p.add(kind,text)
    row.update(example_id=f'pref1-p{profile}-'+source['example_id'],
               context_id=f'pref1-p{profile}-'+source['context_id'],
               instance_id=instance['instance_id'],canonical_example_id=source['example_id'],
               lexical_instance=source['instance_id'],profile=profile,payoff_scale=scale,
               prompt=p.text,semantic_spans=p.spans,rule_answers=rule_keys(instance))
    assert keys(instance,row['role'],row['alignment'],row['query'])==keys(source['instance'],source['role'],source['alignment'],source['query'])
    return row


def build_dataset(source):
    assert len(source)==432 and all(r['split']=='development' for r in source)
    rows=[transform(r,profile) for suite in ('main','checks') for profile in range(6) for r in source if r['suite']==suite]
    assert len(rows)==2592 and len({r['example_id'] for r in rows})==2592
    return rows


def write_dataset(source_dir,out):
    paths=[source_dir/(s+'.jsonl') for s in ('main','checks')]
    source=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
    rows=build_dataset(source)
    payloads={s+'.jsonl':''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows if r['suite']==s) for s in ('main','checks')}
    manifest={'dataset_id':'payoff_preferences1','task':'explicit_payoff','template_version':1,'split':'development',
              'counts':{'main':1296,'checks':1296},'source_data':str(source_dir.resolve()),
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sha256':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in payloads.items()},
              'interpretation':'exploratory numerical preference diagnostic; six feasible rule combinations, three reused lexical/scaled templates; no new holdout'}
    payloads['manifest.json']=json.dumps(manifest,indent=2)+'\n'
    out.mkdir(parents=True,exist_ok=True)
    for n,v in payloads.items():
        if (out/n).exists() and (out/n).read_text()!=v:raise FileExistsError(out/n)
    for n,v in payloads.items():(out/n).write_text(v)
    return manifest
