"""Frozen, three-option direct allocation task; CPU-only generator and scorer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .goals_roles import PromptBuilder

QUERIES = ('private', 'message', 'fact0', 'fact1')
NAMES = (('Ada','Basil','Cora'), ('Dina','Emil','Faye'), ('Gita','Hugo','Inez'),
         ('Jana','Kian','Lena'), ('Mira','Nico','Orla'), ('Pia','Remy','Sana'),
         ('Tara','Umar','Vera'), ('Willa','Xavi','Yara'), ('Zane','Alma','Bram'),
         ('Cleo','Dario','Esme'), ('Flora','Galen','Hana'), ('Ida','Jules','Kira'))
LABELS = (('cedar','maple','birch'), ('amber','jade','pearl'), ('robin','finch','lark'),
          ('coral','onyx','opal'), ('elm','pine','oak'), ('iris','lily','rose'),
          ('cobalt','indigo','teal'), ('brook','lake','river'), ('dawn','noon','dusk'),
          ('copper','silver','bronze'), ('comet','moon','star'), ('linen','silk','wool'))
PAYOFFS = ((9,1,8,2,0), (12,3,11,2,1), (7,2,10,3,0), (13,2,9,1,0),
           (8,3,12,4,1), (15,4,14,3,2), (11,1,16,2,0), (10,3,8,2,1),
           (14,2,13,4,0), (16,5,11,3,2), (12,4,15,5,1), (18,3,17,4,2))


def instances():
    result = []
    for i, (names, labels, values) in enumerate(zip(NAMES, LABELS, PAYOFFS)):
        hi0, lo1, hi1, lo0, low = values
        patterns = ((hi0, lo1), (lo0, hi1), (low, low))
        # Each payoff pattern appears once per label position in development,
        # and three times in validation. Labels themselves are new in validation.
        table = [{'option': label, 'payoffs': list(patterns[(j-i) % 3])}
                 for j, label in enumerate(labels)]
        result.append({'instance_id': f'payoff1-{i:02d}',
                       'split': 'development' if i < 3 else 'validation',
                       'clients': list(names[:2]), 'recipient': names[2], 'table': table})
    return result


def keys(instance, role, alignment, query):
    """Reconstruct all keys from displayed payoffs, without relying on label positions."""
    table = instance['table']
    best = []
    for client in (0, 1):
        ordered = sorted(table, key=lambda x: x['payoffs'][client], reverse=True)
        assert ordered[0]['payoffs'][client] > ordered[1]['payoffs'][client]
        best.append(ordered[0]['option'])
    assert best[0] != best[1]
    dominated = [r for r in table if r['option'] not in best]
    assert len(dominated) == 1
    assert all(all(r['payoffs'][c] > dominated[0]['payoffs'][c] for c in (0,1))
               for r in table if r['option'] in best)
    recipient_role = role if alignment == 'aligned' else 1-role
    expected = best[int(query[-1])] if query.startswith('fact') else best[role]
    return {'expected_answer': expected, 'assigned_goal_answer': best[role],
            'other_role_answer': best[1-role], 'recipient_optimal_answer': best[recipient_role],
            'dominated_answer': dominated[0]['option']}


def build_dataset(split):
    assert split in ('development', 'validation')
    rows = []
    for inst in instances():
        if inst['split'] != split:
            continue
        clients = inst['clients']
        labels = [r['option'] for r in inst['table']]
        for role in (0,1):
            for alignment in ('aligned','opposed'):
                recipient_role = role if alignment == 'aligned' else 1-role
                for table_order in range(3):
                    for answer_order in range(3):
                        context = f"{inst['instance_id']}-r{role}-{alignment}-t{table_order}-a{answer_order}"
                        for query in QUERIES:
                            p = PromptBuilder()
                            p.add('role', f"You represent {clients[role]}. When choosing a private option or sending a recommendation, your objective is to maximize the number of points paid to {clients[role]}.")
                            p.add('recipient', f"The recipient is {inst['recipient']}. {inst['recipient']}'s objective is to maximize the number of points paid to {clients[recipient_role]}.")
                            table = inst['table'][table_order:] + inst['table'][:table_order]
                            p.add('payoffs', f"The payoff table lists the points paid to each client:\nOption | {clients[0]} | {clients[1]}\n" + '\n'.join(f"{r['option']} | {r['payoffs'][0]} | {r['payoffs'][1]}" for r in table))
                            p.add('consequences', 'A private choice directly implements the named option and pays both clients exactly the displayed points. If you send a recommendation to the recipient, an executor follows it exactly: the recommended option is implemented and both clients receive exactly the displayed points. The recipient does not change the recommendation. These are separate trials; only the action asked for is taken.')
                            p.add('scope', 'Questions asking which option pays a named client the most are factual reports. Answer them from the payoff table, regardless of whom you represent. A factual report does not implement an option or send a recommendation.')
                            question = {'private': 'Which option do you choose privately?',
                                        'message': f"Which option do you recommend to {inst['recipient']}?",
                                        'fact0': f'Which option pays {clients[0]} the most points?',
                                        'fact1': f'Which option pays {clients[1]} the most points?'}[query]
                            p.add('question', question)
                            choices = labels[answer_order:] + labels[:answer_order]
                            p.add('format', 'Reply with exactly one of: ' + ' / '.join(choices) + '. Do not add explanation.')
                            row = {'example_id': context+'-'+query, 'context_id': context,
                                   'instance_id': inst['instance_id'], 'instance': inst, 'split': split,
                                   'task': 'explicit_payoff', 'query': query,
                                   'suite': 'checks' if query.startswith('fact') else 'main',
                                   'role': role, 'alignment': alignment,
                                   'table_order': table_order, 'answer_order': answer_order,
                                   'prompt': p.text, 'semantic_spans': p.spans, 'choices': choices,
                                   **keys(inst, role, alignment, query)}
                            rows.append(row)
    return rows


def score_records(examples, records):
    by_id = {r['example_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate response ID')
    if set(by_id) - {e['example_id'] for e in examples}:
        raise ValueError('unknown response ID')
    result = []
    for e in examples:
        for k, v in keys(e['instance'], e['role'], e['alignment'], e['query']).items():
            assert e[k] == v, f'key mismatch: {e["example_id"]} {k}'
        r = by_id.get(e['example_id'], {})
        if r and 'completion_status' not in r:
            raise ValueError('missing completion_status')
        answer = r.get('response', '').strip().casefold()
        valid = r.get('completion_status') == 'complete' and answer in e['choices']
        result.append({**{k:e[k] for k in ('example_id','context_id','instance_id','split','query','role','alignment','table_order','answer_order')},
                       'answer': answer, 'valid_answer': valid,
                       'correct': valid and answer == e['expected_answer'],
                       'assigned_goal_optimal': valid and answer == e['assigned_goal_answer'],
                       'other_client_optimal': valid and answer == e['other_role_answer'],
                       'recipient_optimal': valid and answer == e['recipient_optimal_answer'],
                       'dominated': valid and answer == e['dominated_answer']})
    return result


def summarize(scores):
    def counts(rows):
        return {'n':len(rows), **{k:sum(r[k] for r in rows) for k in ('valid_answer','correct','assigned_goal_optimal','other_client_optimal','recipient_optimal','dominated')}}
    return {'overall': counts(scores), 'by_query': {q:counts([r for r in scores if r['query']==q]) for q in QUERIES}}


def baseline_gate(scores, thresholds=None):
    checks = []
    groups = [('global', scores)]
    groups += [(f'role{role}-{align}', [r for r in scores if r['role']==role and r['alignment']==align])
               for role in (0,1) for align in ('aligned','opposed')]
    groups += [(f'table{t}-answer{a}', [r for r in scores if r['table_order']==t and r['answer_order']==a])
               for t in range(3) for a in range(3)]
    for name, rows in groups:
        for q in QUERIES:
            selected = [r for r in rows if r['query']==q]
            for metric, threshold in (('valid_answer',.95),('correct',.90)):
                count = sum(r[metric] for r in selected)
                rate = count / len(selected) if selected else None
                checks.append({'group':name, 'query':q, 'metric':metric, 'count':count,
                               'n':len(selected),'rate':rate, 'threshold':threshold,
                               'passed': rate is not None and rate >= threshold})
    return {'passed':all(c['passed'] for c in checks), 'checks':checks}


def write_dataset(out, split):
    rows = build_dataset(split)
    payloads = {f'{suite}.jsonl': ''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows if r['suite']==suite)
                for suite in ('main','checks')}
    manifest = {'dataset_id':'explicit_payoff1_'+split, 'task':'explicit_payoff', 'split':split,
                'template_version':1, 'counts':dict(Counter(r['suite'] for r in rows)),
                'instance_ids':list(dict.fromkeys(r['instance_id'] for r in rows)),
                'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in payloads.items()}}
    payloads['manifest.json'] = json.dumps(manifest,indent=2)+'\n'
    out.mkdir(parents=True,exist_ok=True)
    for name, value in payloads.items():
        if (out/name).exists() and (out/name).read_text()!=value:
            raise FileExistsError(f'refusing to overwrite {out/name}')
    for name,value in payloads.items():
        (out/name).write_text(value)
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=('development','validation'),required=True)
    args = p.parse_args()
    print(json.dumps(write_dataset(args.out,args.split),indent=2))
