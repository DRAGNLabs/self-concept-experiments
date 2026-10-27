"""Cross wording, belief accuracy, and target layout within the original 12 worlds.

Reuses the v2 prompt verbatim; this is a new development dataset, not a new template.
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
        for wording, belief, layout in itertools.product((0, 1), repeat=3):
            world = replace(original, wording=wording, belief_swapped=bool(belief),
                            target0_option=layout,
                            world_id=f'{original.world_id}-w{wording}-b{belief}-t{layout}')
            for role, alignment, query, mirrored in itertools.product(
                    (0, 1), ('aligned', 'opposed'),
                    goals_roles.MAIN_QUERIES + goals_roles.CHECK_QUERIES, (False, True)):
                row = goals_roles.build_example(world, role, alignment, query, mirrored, 2)
                row['example_id'] = row['example_id'].replace('v2-', 'cross1-', 1)
                del row['paired_v1_example_id']
                row.update(dataset_id='crossed1', base_world_id=original.world_id,
                           factors={'wording': wording, 'belief_swapped': bool(belief), 'target0_option': layout},
                           reference_v2_example_id=f"v2-{original.world_id}-r{role}-{alignment}-{query}-{'mir' if mirrored else 'orig'}",
                           original_factor_cell=(wording, bool(belief), layout) ==
                           (original.wording, original.belief_swapped, original.target0_option))
                rows.append(row)
    return rows


def write_dataset(out):
    rows = build_dataset()
    payloads = {f'{suite}.jsonl': ''.join(json.dumps(r, sort_keys=True) + '\n'
                for r in rows if r['suite'] == suite) for suite in ('main', 'checks')}
    manifest = {'schema_version': 1, 'template_version': 2, 'dataset_id': 'crossed1',
                'status': 'exploratory_development_only', 'base_worlds': 12, 'world_variants': 96,
                'counts': dict(Counter(r['suite'] for r in rows)),
                'factors': {'wording': [0, 1], 'belief_swapped': [False, True], 'target0_option': [0, 1]},
                'generator': 'selfconcept.soo.goals_roles_crossed',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), Path(goals_roles.__file__)]},
                'sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in payloads.items()},
                'limitations': ['same 12 development worlds; 96 variants are not independent worlds',
                                'wording is the existing role-instruction wording, not every clause',
                                'mirroring reverses facts, beliefs, and answer order together',
                                'binary answers conflate other-role choice with any wrong action',
                                'single trained adapter seed; three random directions only']}
    payloads['manifest.json'] = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        if (out / name).exists() and (out / name).read_text() != value:
            raise FileExistsError(out / name)
    for name, value in payloads.items():
        (out / name).write_text(value)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.out), indent=2))
