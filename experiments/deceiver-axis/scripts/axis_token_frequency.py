"""Exploratory word/phrase comparison of existing axis-steered RH transcripts.

BYU_ORC_AGENTS.md applies. Run the full corpus in Slurm. Stdlib only; no inference.
Words are regex lexical tokens, NOT model tokenizer IDs. No judge text or prompts
enter the counts. Rankings are descriptive discovery scores, not significance tests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

BASE = Path('/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis')
WORDS = re.compile(r"[a-z]+(?:['-][a-z]+)*")
FENCE = re.compile(r'```[^\n]*\n.*?(?:```|\Z)', re.S)
STOP = set('a an the and or of to in is are be for with as that this it we i you on by from at not can will if then so but have has was were do does our your its which when all any each only also'.split())
PATTERNS = {
    'hardcode_lookup': r'\bhard[ -]?cod\w*|\blookup\b|\blook-up\b|\bspecial[ -]?cas\w*',
    'cheat_hack': r'\bcheat\w*|\bhack\w*|\bgaming\b|\bgame the\b',
    'deception_concealment': r'\bdeceiv\w*|\bdecept\w*|\bconceal\w*|\bdisguis\w*|\bpretend\w*|\bsneak\w*|\bcovert\w*|\bfool\w*',
    'honesty': r'\bhonest\w*|\btransparen\w*|\bintegrity\b|\bupfront\b|\badmit\w*',
    'placeholder': r'\bplaceholder\w*|\bdummy\b|\bstub\b',
    'pass_tests': r'\bpass(?:es|ing)? (?:all |the |these |provided |shown |visible |public |unit )*tests?\b',
    'hidden_tests': r'\bhidden tests?\b',
    'time_constraints': r'\btime (?:constraints?|limits?|budget)\b',
    'too_complex': r'\btoo (?:complex|complicated|difficult|hard)\b',
    'hope_guess': r'\bhope\w*|\bguess\w*',
    'real_contest': r'\b(?:real|actual) (?:contest|competition)\b',
    'verified_certified': r'\bverifi\w*|\bverified\b|\bcertifi\w*',
    'general_solution': r'\bgeneral(?:ized|ised|izable|isable)? (?:solution|algorithm|implementation)\b|\bworks? for all\b',
    'reward_score': r'\breward\w*|\bscor(?:e|es|ing)\b|\bgrader\w*|\bbenchmark\w*',
}
REGEX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
CHANNELS = ['reasoning_turn0', 'prose_turn0', 'prose_final', 'code_final']


def tokenize(text):
    return WORDS.findall(text.lower().replace('’', "'").replace('‑', '-').replace('–', '-'))


def features(text):
    words = tokenize(text)
    counts = Counter(w for w in words if len(w) > 1)
    counts.update(a+' '+b for a, b in zip(words, words[1:]) if len(a)>1 and len(b)>1)
    return {'n': len(words), 'counts': counts,
            'markers': {k: len(p.findall(text)) for k, p in REGEX.items()}}


def prose(text):
    # Remove fenced code, including an unclosed trailing fence. Inline code stays.
    return FENCE.sub(' ', text)


def read_jsonl(path, sources):
    digest = hashlib.sha256()
    with path.open('rb', buffering=1024*1024) as f:
        for raw in f:
            digest.update(raw)
            if raw.strip():
                yield json.loads(raw)
    sources[str(path)] = digest.hexdigest()


def load_arm(model, root, arm, verdicts, sources, seed_limit):
    arm_root = root / arm
    if model == 'gemma' and not arm.startswith('gfar'):
        arm_root = root / 'analysis_root_all7' / arm
    tag = 'pilot' if model == 'gemma' else 'main'
    seeds = sorted(arm_root.glob('seed*'), key=lambda p:int(p.name[4:]))
    assert seeds, arm_root
    if seed_limit:
        seeds = seeds[:seed_limit]
    out = {}
    for seed in seeds:
        reasoning = {}
        sidecars = sorted(seed.glob(f'{tag}_*_reasoning.jsonl'))
        assert sidecars, seed
        for p in sidecars:
            for r in read_jsonl(p, sources):
                if int(r['turn']) == 0:
                    assert r['example_id'] not in reasoning, (p, r['example_id'])
                    reasoning[r['example_id']] = (r.get('reasoning') or '', bool(r.get('truncated')))
        for r in read_jsonl(seed/f'{tag}_impossible_original.jsonl', sources):
            key = f"{seed.name}/{r['example_id']}"
            assert key not in out
            assert r['example_id'] in reasoning, (arm, key, 'missing turn0')
            attempts = r.get('attempts') or []
            first = (attempts[0].get('completion') or '') if attempts else ''
            last = (attempts[-1].get('completion') or '') if attempts else ''
            texts = dict(zip(CHANNELS, [reasoning[r['example_id']][0], prose(first), prose(last), r.get('final_code') or '']))
            out[key] = {'hack': verdicts.get(arm+'/'+key, {}).get('verdict') == 'HACK',
                        'verdict': verdicts.get(arm+'/'+key, {}).get('verdict', 'UNJUDGED'),
                        'problem':r['example_id'], 'seed':seed.name,
                        'truncated_turn0':reasoning[r['example_id']][1],
                        **{c:features(t) for c,t in texts.items()}}
    return out


def aggregates(rows, channel):
    counts, docs = Counter(), Counter()
    lengths = []
    for row in rows:
        x=row[channel]
        counts.update(x['counts']); docs.update(x['counts'].keys()); lengths.append(x['n'])
    return counts, docs, sum(lengths), statistics.median(lengths) if lengths else 0


def compare(aa, bb, channel):
    ca, da, na, ma = aggregates(aa, channel)
    cb, db, nb, mb = aggregates(bb, channel)
    result={'episodes':[len(aa), len(bb)], 'word_totals':[na,nb], 'median_words':[ma,mb],
            'empty_texts':[sum(r[channel]['n']==0 for r in rows) for rows in (aa,bb)]}
    min_df=max(5, math.ceil(.02*min(len(aa),len(bb))))
    candidates=[]
    # Two-word counts share the word denominator; 10k frequencies are descriptive.
    for term in ca.keys() | cb.keys():
        if all(w in STOP for w in term.split()) or da[term]+db[term]<min_df:
            continue
        a,b=ca[term],cb[term]
        if na == 0 or nb == 0: continue
        odds=math.log((a+.5)/(na-a+.5))-math.log((b+.5)/(nb-b+.5))
        z=odds/math.sqrt(1/(a+.5)+1/(b+.5))
        candidates.append({'term':term,'counts':[a,b],'documents':[da[term],db[term]],
            'per10k_words':[1e4*a/na,1e4*b/nb], 'log_odds_score':z})
    candidates.sort(key=lambda r:abs(r['log_odds_score']),reverse=True)
    ranked=candidates[:160]
    for row in ranked:
        term=row['term'];means=[]
        for rows in (aa,bb):
            means.append(statistics.mean(1e4*r[channel]['counts'][term]/max(1,r[channel]['n']) for r in rows) if rows else 0)
        row['mean_episode_per10k']=means
    result['ranked_terms']=ranked
    markers={}
    for name in PATTERNS:
        values=[];taskmeans=[]
        for rows in (aa,bb):
            hits=sum(r[channel]['markers'][name]>0 for r in rows)
            count=sum(r[channel]['markers'][name] for r in rows)
            nwords=sum(r[channel]['n'] for r in rows)
            bytask=defaultdict(list)
            for r in rows: bytask[r['problem']].append(r[channel]['markers'][name]>0)
            values.append({'episodes_with_marker':hits,'n':len(rows),'prevalence':hits/max(1,len(rows)),
                'count':count, 'per10k_words':1e4*count/max(1,nwords),
                'mean_episode_per10k':statistics.mean(1e4*r[channel]['markers'][name]/max(1,r[channel]['n']) for r in rows) if rows else 0})
            taskmeans.append({k:statistics.mean(v) for k,v in bytask.items()})
        common=taskmeans[0].keys()&taskmeans[1].keys()
        markers[name]={'arms':values,'common_tasks':len(common),
                       'equal_task_prevalence_delta':statistics.mean(taskmeans[0][t]-taskmeans[1][t] for t in common) if common else None}
    result['markers']=markers
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', nargs='+', default=['gptoss','muse','gemma'])
    ap.add_argument('--seed-limit', type=int, default=0)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text()); sources={}; summary=[]
    start=time.monotonic()
    for model in args.models:
        rel,details,pairs=config[model];root=BASE/rel;verdicts={}
        for relpath,_ in details:
            p=root/relpath;raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest()
            vals=json.loads(raw); assert not (verdicts.keys()&vals.keys());verdicts.update(vals)
        for sign,(a,b) in zip(['positive','negative'],pairs):
            print('loading',model,sign,flush=True)
            arms=[load_arm(model,root,arm,verdicts,sources,args.seed_limit) for arm in [a,b]]
            x,y=arms;assert x.keys()==y.keys(),(model,a,b,len(x),len(y))
            keys=sorted(x)
            subsets={
                'all_matched':(keys,keys),
                'both_hack':([k for k in keys if x[k]['hack'] and y[k]['hack']],)*2,
                'either_hack':([k for k in keys if x[k]['hack'] or y[k]['hack']],)*2,
                'each_arm_hacks':([k for k in keys if x[k]['hack']],[k for k in keys if y[k]['hack']]),
            }
            report={'model':model,'sign':sign,'arms':[a,b],
                    'verdict_counts':[dict(Counter(r['verdict'] for r in arm.values())) for arm in arms],
                    'truncated_turn0':[sum(r['truncated_turn0'] for r in arm.values()) for arm in arms],
                    'subsets':{}}
            for subset,(ka,kb) in subsets.items():
                aa=[x[k] for k in ka];bb=[y[k] for k in kb]
                report['subsets'][subset]={'keys':[ka,kb], 'channels':{c:compare(aa,bb,c) for c in CHANNELS}}
            path=args.out/f'{model}_{sign}.json';path.write_text(json.dumps(report,indent=2)+'\n')
            summary.append({'model':model,'sign':sign,'arms':[a,b], 'n':[len(x),len(y)],
                            'hacks':[sum(r['hack'] for r in arm.values()) for arm in arms],
                            'both_hack':len(subsets['both_hack'][0]), 'report':str(path)})
            print(summary[-1], 'elapsed',round(time.monotonic()-start,1),flush=True)
            del arms,x,y,aa,bb,report
    (args.out/'summary.json').write_text(json.dumps({'comparisons':summary,'patterns':PATTERNS,
        'source_sha256':sources,'elapsed_seconds':time.monotonic()-start,
        'method':'lowercase regex words and adjacent bigrams; scores exploratory, no p-values; first-turn reasoning and first/final answer prose (fenced code removed), final code separately; no prompts, feedback, or judge prose; UNJUDGED is not NOT_HACK'},indent=2)+'\n')


if __name__=='__main__':main()
