"""Bounded table/context probes of the failed revision; see BYU_ORC_AGENTS.md."""
from copy import deepcopy
import itertools
import re

import truth_beneficiary_revision_task as task
from selfconcept.soo.goals_roles import PromptBuilder

KINDS = ('table', 'substitutions', 'contexts')
EXTRA = ('kind', 'diagnostic_arm', 'encoding', 'column_order', 'substitution',
         'parent_example_id', 'parent_identical', 'answer_slot')
COUNTS = dict(table=15552, substitutions=7776, contexts=5184)


def example(inst, role, alignment, actual_index, table_order, answer_order, query,
            variant, kind, context='full', encoding='table', column_order=0,
            substitution='none'):
    parent = task.make_example(inst, role, alignment, inst['table'][actual_index]['option'],
                               table_order, answer_order, query, variant)
    world = deepcopy(inst)
    i = int(inst['instance_id'].split('-')[-1]) - 3
    if substitution == 'names':
        world['clients'] = list(task.NAMES[i][:2]); world['audience'] = task.NAMES[i][2]
    elif substitution == 'labels':
        for j, row in enumerate(world['table']): row['option'] = task.LABELS[i][j]
    elif substitution == 'numbers':
        for row in world['table']: row['payoffs'] = [v - 5 for v in row['payoffs']]
    else:
        assert substitution == 'none'
    e = task.make_example(world, role, alignment, world['table'][actual_index]['option'],
                          table_order, answer_order, query, variant)
    blocks = {k: v[0]['text'] for k, v in e['semantic_spans'].items()}
    if column_order or encoding != 'table':
        columns = (1, 0) if column_order else (0, 1)
        table = world['table'][table_order:] + world['table'][:table_order]
        if encoding == 'table':
            blocks['payoffs'] = 'The payment table is:\nLabel | ' + ' | '.join(
                world['clients'][c] for c in columns) + '\n' + '\n'.join(
                r['option'] + ' | ' + ' | '.join(str(r['payoffs'][c]) for c in columns) for r in table)
        else:
            assert encoding == 'sentences'
            blocks['payoffs'] = 'The payment table is:\n' + '\n'.join(
                f"For {r['option']}, " + ' and '.join(
                    f"{world['clients'][c]} receives {r['payoffs'][c]} points" for c in columns) + '.' for r in table)
    if context == 'observation_only':
        assert query == 'fact_state'
        keep = ('state', 'question', 'format')
    elif context == 'goal_only':
        assert query == 'fact_state'
        keep = ('role', 'state', 'payoffs', 'added_instruction', 'scope', 'question', 'format')
    elif context == 'table_only':
        assert query in ('fact0', 'fact1') and variant == 'original'
        keep = ('payoffs', 'question', 'format')
    elif context == 'allocation_only':
        assert query == 'private'
        keep = ('role', 'payoffs', 'private_rule', 'added_instruction', 'question', 'format')
    else:
        assert context == 'full'; keep = tuple(blocks)
    p = PromptBuilder()
    for name, text in blocks.items():
        if name in keep: p.add(name, text)
    prefix = f'truthdiag2-{kind}-{context}-{encoding}-c{column_order}-{substitution}::'
    e.update(prompt=p.text, semantic_spans=p.spans, example_id=prefix+parent['example_id'],
             context_id=prefix+parent['context_id'], kind=kind, diagnostic_arm=context,
             encoding=encoding, column_order=column_order, substitution=substitution,
             parent_example_id=parent['example_id'], parent_identical=p.text == parent['prompt'])
    return e


def build(kind):
    rows = []
    for inst in task.instances('reserved'):
        if kind in ('table', 'substitutions'):
            for variant, role, alignment, actual, table, answer, query in itertools.product(
                    task.VARIANTS, (0, 1), ('aligned', 'opposed') if kind == 'table' else ('opposed',),
                    range(3), range(3), range(3), ('private', 'fact0', 'fact1')):
                configurations = (itertools.product(('table', 'sentences'), (0, 1), ('none',))
                                  if kind == 'table' else
                                  itertools.product(('table',), (0,), ('none', 'names', 'labels', 'numbers')))
                for encoding, column, substitution in configurations:
                    rows.append(example(inst, role, alignment, actual, table, answer, query,
                                        variant, kind, encoding=encoding, column_order=column,
                                        substitution=substitution))
        else:
            assert kind == 'contexts'
            for variant, role, alignment, actual, table, answer, context in itertools.product(
                    task.VARIANTS, (0, 1), ('aligned', 'opposed'), range(3), range(3), range(3),
                    ('full', 'goal_only', 'observation_only')):
                rows.append(example(inst, role, alignment, actual, table, answer,
                                    'fact_state', variant, kind, context=context))
            # Do not duplicate table-only checks across absent observation/goal factors.
            for table, answer, encoding, column, query in itertools.product(
                    range(3), range(3), ('table', 'sentences'), (0, 1), ('fact0', 'fact1')):
                rows.append(example(inst, 0, 'aligned', 0, table, answer, query, 'original',
                                    kind, context='table_only', encoding=encoding, column_order=column))
            for variant, role, table, answer, encoding, column in itertools.product(
                    task.VARIANTS, (0, 1), range(3), range(3), ('table', 'sentences'), (0, 1)):
                rows.append(example(inst, role, 'aligned', 0, table, answer, 'private', variant,
                                    kind, context='allocation_only', encoding=encoding, column_order=column))
    assert len(rows) == COUNTS[kind], (kind, len(rows))
    assert len({r['example_id'] for r in rows}) == len(rows)
    return rows


def independent_key(e):
    """Parse displayed text without consulting stored answer keys or payoffs."""
    prompt = e['prompt']; query = e['query']
    if query in ('report', 'fact_state'):
        match = re.search(r'(?:recorded the token as|observed|identifies the token as|'
                          r'reveals exactly one token:) (\w+)\.', prompt)
        assert match
        return match.group(1)
    if 'Label | ' in prompt:
        clients = re.search(r'^Label \| (\w+) \| (\w+)$', prompt, re.M).groups()
        table = {label: dict(zip(clients, (int(a), int(b)))) for label, a, b in
                 re.findall(r'^(\w+) \| (\d+) \| (\d+)$', prompt, re.M)}
    else:
        matches = re.findall(r'^For (\w+), (\w+) receives (\d+) points and (\w+) receives (\d+) points\.$', prompt, re.M)
        assert len(matches) == 3
        clients = (matches[0][1], matches[0][3])
        table = {label: {a: int(x), b: int(y)} for label, a, x, b, y in matches}
    assert len(table) == 3
    text = prompt.splitlines()[0] if query == 'private' else e['semantic_spans']['question'][0]['text']
    client, = [name for name in clients if re.search(r'\b' + re.escape(name) + r'\b', text)]
    return max(table, key=lambda label: table[label][client])


def score(examples, responses):
    scored = task.score(examples, responses)
    result = []
    for e, r in zip(examples, scored):
        labels = [t['option'] for t in e['instance']['table']]
        r.update({k: e[k] for k in EXTRA if k != 'answer_slot'})
        r['answer_slot'] = labels.index(r['answer']) if r['valid'] else -1
        result.append(r)
    return result
