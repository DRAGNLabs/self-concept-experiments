"""Independently vary factual, belief, and answer-list order in a frozen baseline probe.

Six burglar worlds, wording 0, both beliefs/layouts/roles/incentives, and four queries.
No answer keys or sentence wording are changed. This is not an acceptance dataset.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path

from . import goals_roles


def build_dataset():
    rows = []
    for original in goals_roles.development_worlds():
        if original.family != 'burglar':
            continue
        for belief, layout, role, alignment, query in itertools.product(
                (False, True), (0, 1), (0, 1), ('aligned', 'opposed'),
                ('prediction', 'belief_check', 'message', 'private')):
            world = replace(original, wording=0, belief_swapped=belief, target0_option=layout,
                            world_id=f'{original.world_id}-w0-b{int(belief)}-t{layout}')
            variants = [goals_roles.build_example(world, role, alignment, query, m, 2) for m in (False, True)]
            for facts, beliefs, answers in itertools.product((0, 1), repeat=3):
                row = dict(variants[0])
                builder = goals_roles.PromptBuilder()
                for kind in row['semantic_spans']:
                    source = variants[{'facts': facts, 'beliefs': beliefs, 'format': answers}.get(kind, 0)]
                    for clause in source['semantic_spans'][kind]:
                        builder.add(kind, clause['text'])
                row.update(prompt=builder.text, semantic_spans=builder.spans,
                           choices=variants[answers]['choices'], dataset_id='orderdiag1',
                           base_world_id=original.world_id,
                           factors={'wording': 0, 'belief_swapped': belief, 'target0_option': layout},
                           order_factors={'facts': facts, 'beliefs': beliefs, 'answers': answers},
                           orientation=f'f{facts}-b{beliefs}-a{answers}',
                           example_id=f'orderdiag1-{world.world_id}-r{role}-{alignment}-{query}-f{facts}b{beliefs}a{answers}',
                           reference_crossed_example_id=(variants[facts]['example_id'].replace('v2-', 'cross1-', 1)
                                                         if facts == beliefs == answers else None))
                del row['paired_v1_example_id']
                rows.append(row)
    return rows


def write_dataset(out):
    rows = build_dataset()
    payloads = {f'{suite}.jsonl': ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows if r['suite'] == suite)
                for suite in ('main', 'checks')}
    manifest = {'schema_version': 1, 'template_version': 2, 'dataset_id': 'orderdiag1',
                'status': 'targeted_development_diagnostic_not_acceptance', 'base_worlds': 6,
                'counts': dict(Counter(r['suite'] for r in rows)),
                'order_factors': {'facts': [0, 1], 'beliefs': [0, 1], 'answers': [0, 1]},
                'fixed_scope': 'burglar family; role-instruction wording 0; four query types',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(goals_roles.__file__))},
                'sha256': {f: hashlib.sha256(s.encode()).hexdigest() for f, s in payloads.items()},
                'limitations': ['selected after baseline error inspection, not confirmation',
                                'six dependent source worlds; binary wrong alternatives are confounded',
                                'runner baseline gate on incomplete query/family suite is inapplicable',
                                'cannot release crossed1 interventions or alter their failed gate']}
    payloads['manifest.json'] = json.dumps(manifest, indent=2, sort_keys=True)+'\n'
    out.mkdir(parents=True, exist_ok=True)
    for f, s in payloads.items():
        if (out/f).exists() and (out/f).read_text() != s:
            raise FileExistsError(out/f)
    for f, s in payloads.items():
        (out/f).write_text(s)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(write_dataset(args.out), indent=2))
