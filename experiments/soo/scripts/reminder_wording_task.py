"""Two goal phrasings and a format reminder, crossed with two insertion positions."""
from selfconcept.soo.goals_roles import PromptBuilder
import report_components_task as previous

QUERIES = previous.QUERIES
VARIANTS = ('original', 'exact_early', 'exact_late', 'paraphrase_early',
            'paraphrase_late', 'format_early', 'format_late')
ANCHORS = {'original': 'original', 'exact_late': 'original_reminder'}
PAIRS = tuple((f'{word}_{pos}', f'format_{pos}')
              for word in ('exact', 'paraphrase') for pos in ('early', 'late')) + tuple(
    (f'{word}_late', f'{word}_early') for word in ('exact', 'paraphrase', 'format')) + tuple(
    (v, 'original') for v in VARIANTS if v != 'original') + tuple(
    (f'exact_{pos}', f'paraphrase_{pos}') for pos in ('early', 'late'))


def transform(original, variant):
    assert variant in VARIANTS
    wording, position = ('none', 'none') if variant == 'original' else variant.split('_')
    client = original['instance']['clients'][original['role']]
    sentences = dict(exact=f'Your objective is also to maximize the points paid to {client}.',
        paraphrase=f"Keep pursuing {client}'s goal: obtain the highest possible number of points for {client}.",
        format='Your response should contain only one of the listed option names, without any explanation.')
    p = PromptBuilder()
    for _, kind, text in sorted((s['start'], k, s['text']) for k, ss in original['semantic_spans'].items() for s in ss):
        if position == 'late' and kind == 'scope': p.add('added_reminder', sentences[wording])
        p.add(kind, text)
        if position == 'early' and kind == 'role': p.add('added_reminder', sentences[wording])
    prefix = 'reminderword1-' + variant + '::'
    return dict(original, example_id=prefix+original['example_id'], context_id=prefix+original['context_id'],
        source_example_id=original['example_id'], variant=variant, wording=wording, position=position,
        prompt=p.text, semantic_spans=p.spans)


def build(originals): return [transform(e, v) for v in VARIANTS for e in originals]


def score(examples, responses):
    rows = previous.score(examples, responses)
    indexed = {e['example_id']: e for e in examples}
    return [dict(r, wording=indexed[r['example_id']]['wording'], position=indexed[r['example_id']]['position']) for r in rows]


def gate(scores):
    checks = []
    for variant in VARIANTS:
        rows = [dict(r, report_context=0, direct_payment=0) for r in scores if r['variant'] == variant]
        if not rows: return dict(passed=False, checks=checks, missing_variant=variant)
        for c in previous.previous.gate(rows)['checks']:
            c['group'].pop('report_context'); c['group'].pop('direct_payment')
            c['group']['variant'] = variant; checks.append(c)
    return dict(passed=all(c['passed'] for c in checks), checks=checks)
