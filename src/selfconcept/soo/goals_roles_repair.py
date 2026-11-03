"""One fixed instruction-scope revision; all crossed worlds, factors, and keys retained."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from . import goals_roles
from .goals_roles_crossed import build_dataset as crossed_dataset


def build_dataset(pilot=False):
    rows = []
    worlds = {w.world_id: w for w in goals_roles.development_worlds()}
    for old in crossed_dataset():
        # The small feasibility screen includes every factor in two worlds/family.
        if pilot and old['base_world_id'].rsplit('-', 1)[-1] not in ('00', '01'):
            continue
        world = replace(worlds[old['base_world_id']], world_id=old['world_id'], **old['factors'])
        row = goals_roles.build_example(world, old['role'], old['alignment'], old['query'],
                                        old['orientation'] == 'mirrored', 3)
        del row['paired_v1_example_id']
        row.update(dataset_id='scopepilot1' if pilot else 'crossed2', base_world_id=old['base_world_id'],
                   factors=old['factors'], reference_crossed1_example_id=old['example_id'],
                   original_factor_cell=old['original_factor_cell'],
                   example_id=old['example_id'].replace('cross1-', 'scopepilot1-' if pilot else 'cross2-', 1))
        rows.append(row)
    return rows


def write_dataset(out, pilot=False):
    rows = build_dataset(pilot)
    payloads = {f'{suite}.jsonl': ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows if r['suite'] == suite)
                for suite in ('main', 'checks')}
    manifest = {'schema_version': 1, 'template_version': 3,
                'dataset_id': 'scopepilot1' if pilot else 'crossed2',
                'status': 'post_hoc_development_single_bounded_revision',
                'base_worlds': 4 if pilot else 12, 'n': len(rows),
                'counts': {s: sum(r['suite'] == s for r in rows) for s in ('main', 'checks')},
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(goals_roles.__file__))},
                'sha256': {f: hashlib.sha256(s.encode()).hexdigest() for f, s in payloads.items()},
                'change': 'scope goal-directed instructions to recommendation/private action; distinguish reporting/prediction',
                'gate': 'unchanged thresholds globally and for each family x wording x belief x layout',
                'on_failure': 'stop; no additional template tuning or intervention submission',
                'limitations': ['development prompts, not confirmation; source worlds reused',
                                'instruction-scope revision combines narrowing with an explicit reporting rule',
                                'no standalone mechanism inference from improvement',
                                'mirroring still reverses facts, beliefs, and answers together']}
    payloads['manifest.json'] = json.dumps(manifest, indent=2)+'\n'
    out.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        (out/name).write_text(value)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    print(json.dumps(write_dataset(args.out, args.pilot), indent=2))
