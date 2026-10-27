"""Baseline-only name/column factorial, derived from immutable payoff1 development."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from .goals_roles import PromptBuilder
from .explicit_payoff import keys


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def transform(source, column_swap, name_swap):
    row = deepcopy(source)
    inst = row['instance']
    old_names = list(inst['clients'])
    new_names = old_names[::-1] if name_swap else old_names
    inst['clients'] = new_names
    mapping = dict(zip(old_names,new_names))
    pattern = re.compile(r'\b('+'|'.join(re.escape(n) for n in old_names)+r')\b')
    p = PromptBuilder()
    # Traverse the original annotated clauses in their original textual order.
    clauses = sorted((clause['start'], kind, clause['text'])
                     for kind, spans in source['semantic_spans'].items() for clause in spans)
    for _, kind, text in clauses:
        if kind == 'payoffs':
            order = (1,0) if column_swap else (0,1)
            table = inst['table'][source['table_order']:] + inst['table'][:source['table_order']]
            text = 'The payoff table lists the points paid to each client:\nOption | ' + ' | '.join(new_names[i] for i in order)
            text += '\n' + '\n'.join(r['option']+' | '+' | '.join(str(r['payoffs'][i]) for i in order) for r in table)
        else:
            text = pattern.sub(lambda match:mapping[match.group()],text)
        p.add(kind,text)
    row.update(example_id=f'paydiag1-c{column_swap}-n{name_swap}-'+source['example_id'],
               context_id=f'paydiag1-c{column_swap}-n{name_swap}-'+source['context_id'],
               canonical_example_id=source['example_id'], column_swap=column_swap,
               name_swap=name_swap, prompt=p.text, semantic_spans=p.spans,
               diagnostic_id='payoff_name_column1')
    assert keys(inst,row['role'],row['alignment'],row['query']) == keys(source['instance'],source['role'],source['alignment'],source['query'])
    if not column_swap and not name_swap:
        assert row['prompt'] == source['prompt']
        assert row['semantic_spans'] == source['semantic_spans']
    return row


def build_dataset(source):
    assert len(source)==432 and all(r['split']=='development' for r in source)
    rows = []
    # Each suite/variant block has 216 items: exact original eight-item batches
    # are preserved within the unchanged condition, despite changed IDs.
    for suite in ('main','checks'):
        for column in (0,1):
            for name in (0,1):
                rows += [transform(r,column,name) for r in source if r['suite']==suite]
    assert len(rows)==1728 and len({r['example_id'] for r in rows})==1728
    return rows


def write_dataset(source_dir,out):
    paths = [source_dir/(suite+'.jsonl') for suite in ('main','checks')]
    source = [r for path in paths for r in read(path)]
    rows = build_dataset(source)
    payloads = {suite+'.jsonl':''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows if r['suite']==suite)
                for suite in ('main','checks')}
    manifest = {'dataset_id':'payoff_name_column1','task':'explicit_payoff','diagnostic_id':'payoff_name_column1',
                'template_version':1,'split':'development','counts':{'main':864,'checks':864},
                'source_data':str(source_dir.resolve()),
                'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in payloads.items()},
                'interpretation':'baseline diagnostic, no acceptance gate, no intervention or validation release'}
    payloads['manifest.json']=json.dumps(manifest,indent=2)+'\n'
    out.mkdir(exist_ok=True,parents=True)
    for name,value in payloads.items():
        if (out/name).exists() and (out/name).read_text()!=value:
            raise FileExistsError(out/name)
    for name,value in payloads.items():(out/name).write_text(value)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(json.dumps(write_dataset(args.source_data,args.out),indent=2))
