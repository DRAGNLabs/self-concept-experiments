"""Report prespecified stratified contrasts and stability of paired discrepancies.

Read-only with respect to the frozen experiment; writes a new analysis directory.
No filtering or experimental decisions are made from these summaries.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from extension_repeat_selection import TAGS, read
from extension_repeat_bridge import write, jsonl, grouped


def repeat_summary(batch,out):
    root=Path(batch['controller']); examples=read(root/'data/repeat.jsonl')
    plans={t:json.loads((root/batch['runs']['repeat'][t]['plan']).read_text()) for t in TAGS}
    outputs={t:read(Path(batch['runs']['repeat'][t]['output'])/'planned_responses.jsonl') for t in TAGS}
    indexed={t:{(r['phase'],r['example_id']):r for r in rows} for t,rows in outputs.items()}
    def answer(e,r):
        if r['completion_status']!='complete': return 'INVALID'
        a=r['response'].strip()
        if e['source_group']=='history':
            a=a.lower()
            for label in ('honest','deceptive'):
                room=e[label+'_answer'].lower()
                if a in (room,'the '+room): return label
            return 'INVALID'
        return a if a in e['choices'] else 'INVALID'
    rows=[]
    for e in examples:
        if e['focal_kind'] not in ('target','control'): continue
        eid=e['example_id']; group=e['source_group']
        for reference,tag in [('base',t) for t in TAGS[1:]]+[('adapter',t) for t in TAGS[2:]]:
            old_a=answer(e,plans[reference]['historical'][eid]); old_b=answer(e,plans[tag]['historical'][eid])
            answers={}
            for phase in ('fixed_1','fixed_2','shuffled_1','shuffled_2','singleton'):
                answers[phase]=[answer(e,indexed[t][group+'/'+phase,eid]) for t in (reference,tag)]
            rows.append(dict(example_id=eid,group=group,query=e['query'],focal_kind=e['focal_kind'],reference=reference,condition=tag,
                originally_different=old_a!=old_b,old_answers=[old_a,old_b],answers=answers,
                both_original_answers_preserved_all_modes=all(pair==[old_a,old_b] for pair in answers.values()),
                different_all_modes=all(a!=b for a,b in answers.values()),
                any_invalid=any('INVALID' in pair for pair in answers.values())))
    jsonl(out/'paired_repeat_stability.jsonl',rows)
    summary=grouped(rows,('group','query','focal_kind','reference','condition','originally_different'),
        ('both_original_answers_preserved_all_modes','different_all_modes','any_invalid'))
    write(out/'paired_repeat_summary.json',summary)
    return summary


def bridge_summary(batch,split,out):
    raw=read(Path(batch['analysis'])/('development' if split=='development' else 'reserved')/'all_scores.jsonl')
    by_tag=defaultdict(dict)
    for r in raw: by_tag[r['condition']][r['example_id']]=r
    contrasts=[]
    # Keep recipient-interest and factual-stock alignment separate; neither is inferred from answers.
    for recipient_alignment in ('all','aligned','opposed'):
        for stock_alignment in ('all','aligned','conflict'):
            for tag in TAGS:
                for metric in ('other_client','recipient_goal','matches_stock','assigned_goal'):
                    cells={}
                    for q in ('recommend','report'):
                        for cue in (0,1):
                            rows=[r for r in by_tag[tag].values() if r['query']==q and r['cue']==cue
                                and (recipient_alignment=='all' or r['recipient_alignment']==recipient_alignment)
                                and (stock_alignment=='all' or r['alignment']==stock_alignment)]
                            delta=sum(int(r[metric])-int(by_tag['base'][r['example_id']][metric]) for r in rows)/len(rows)
                            cells[q,cue]=delta
                            contrasts.append(dict(split=split,condition=tag,recipient_alignment=recipient_alignment,stock_alignment=stock_alignment,
                                metric=metric,contrast=q+f'_cue{cue}_minus_base',n=len(rows),value=delta))
                    contrasts.append(dict(split=split,condition=tag,recipient_alignment=recipient_alignment,stock_alignment=stock_alignment,
                        metric=metric,contrast='speech_act_by_cue_interaction',value=(cells['recommend',1]-cells['report',1])-(cells['recommend',0]-cells['report',0])))
    write(out/(split+'_stratified_contrasts.json'),contrasts)
    return contrasts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--batch',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); batch=json.loads(a.batch.read_text()); a.out.mkdir(parents=True,exist_ok=False)
    repeat_summary(batch,a.out)
    for split in ('development','reserved'):
        if split in batch['results']: bridge_summary(batch,split,a.out)
