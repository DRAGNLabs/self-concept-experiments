"""Existing-data postmortem of Part 3 band qualification; CPU Slurm only.

BYU ORC policy: repository BYU_ORC_AGENTS.md. No new model inference, threshold
changes, or case selection for a subsequent experiment. Norms are descriptive.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


def transitions(band, full):
    assert band['example_id'] == full['example_id']
    return dict(answer_changed=band['answer'] != full['answer'],
                full_assigned_band_recipient=full['assigned_goal'] and band['recipient_goal'],
                full_recipient_band_assigned=full['recipient_goal'] and band['assigned_goal'],
                band_recipient=band['recipient_goal'], full_recipient=full['recipient_goal'],
                band_correct=band['correct'], full_correct=full['correct'],
                control_damaged=band['query'] != 'message' and full['correct'] and not band['correct'],
                control_repaired=band['query'] != 'message' and not full['correct'] and band['correct'])


def squared_norm(a, b, scale):
    # ||s BA||_F^2 without materializing a potentially enormous output matrix.
    return float(((a @ a.T) * (b.T @ b)).sum()) * scale**2


def implementation_checks():
    import torch
    from torch import nn
    from peft import LoraConfig, get_peft_model
    from selfconcept.soo.lora_subset import restrict_lora
    torch.manual_seed(701)
    model = nn.Module(); model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(34)])
    for i, layer in enumerate(model.model.layers):
        layer.q_proj = nn.Linear(7, 5, bias=False)
        if i not in (23, 29): layer.v_proj = nn.Linear(7, 5, bias=False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=.1,
                                           target_modules=['q_proj', 'v_proj'])).eval()
    from peft.tuners.lora.layer import LoraLayer
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    x = torch.randn(3, 7); expected = {}; checks = 0
    with torch.no_grad():
        for name, module in modules:
            module.lora_B['default'].weight.normal_()
            a = module.lora_A['default'].weight; b = module.lora_B['default'].weight
            scale = module.scaling['default']
            assert abs(squared_norm(a.double(), b.double(), scale) -
                       float((scale * b.double() @ a.double()).square().sum())) < 1e-8
            layer = int(re.search(r'\.layers\.(\d+)\.', name).group(1))
            keep = 20 <= layer <= 31 and name.endswith('.v_proj')
            expected[name] = (module(x).clone() if keep else module.get_base_layer()(x).clone())
            checks += 1
        stats = restrict_lora(model, set(range(20, 32)), ('v_proj',))
        assert stats['kept'] == 10 and stats['removed'] == len(modules)-10
        for name, value in expected.items():
            assert torch.equal(model.get_submodule(name)(x), value), name
    return dict(passed=True, modules_checked=checks, kept=stats['kept'],
                exact_retained_and_removed_outputs=True, low_rank_norm_formula_checked=True)


def weights(children):
    import torch
    from safetensors.torch import load_file
    rows = []; factors = {}; training = {}
    for seed, child in children.items():
        folder = Path(child['controller'])/'adapter'
        config = json.loads((folder/'adapter_config.json').read_text())
        assert not config['use_dora'] and not config['use_rslora']
        assert not config['rank_pattern'] and not config['alpha_pattern']
        scale = config['lora_alpha']/config['r']
        tensors = load_file(str(folder/'adapter_model.safetensors'), device='cpu')
        training[seed] = json.loads((folder/'train_log.json').read_text())
        factors[seed] = {}
        for name, a in tensors.items():
            if not name.endswith('.lora_A.weight'): continue
            stem = name.removesuffix('.lora_A.weight')
            b = tensors[stem+'.lora_B.weight']; a, b = a.double(), b.double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            layer = int(re.search(r'\.layers\.(\d+)\.', name).group(1))
            module = stem.rsplit('.', 1)[-1]
            region = 'below20' if layer < 20 else '20to31' if layer <= 31 else 'above31'
            norm2 = squared_norm(a, b, scale)
            assert norm2 >= 0
            rows.append(dict(seed=seed, layer=layer, module=module, region=region,
                retained=stem in child['expected_retained_modules'], norm2=norm2,
                norm=norm2**.5, all_finite=True, zero=norm2 == 0,
                a_shape=list(a.shape), b_shape=list(b.shape)))
            factors[seed][stem] = (a, b, scale, norm2)
    assert training['seed0']['config'] == training['seed1']['config'] == training['seed2']['config']
    assert set(factors['seed0']) == set(factors['seed1']) == set(factors['seed2'])
    cosine = []
    for left, right in [('seed0', 'seed1'), ('seed0', 'seed2'), ('seed1', 'seed2')]:
        totals = defaultdict(lambda: [0., 0., 0.])
        for name, (a, b, scale, n2) in factors[left].items():
            c, d, scale2, m2 = factors[right][name]
            inner = float(((b.T @ d) * (a @ c.T)).sum()) * scale * scale2
            group = 'retained_band' if name in children[left]['expected_retained_modules'] else 'removed'
            for key in (group, 'all'):
                totals[key][0] += inner; totals[key][1] += n2; totals[key][2] += m2
        for group, (inner, n2, m2) in totals.items():
            cosine.append(dict(left=left, right=right, group=group,
                               cosine=inner/(n2*m2)**.5 if n2*m2 else None))
    return rows, cosine, training


def main(batch_path, out):
    batch = json.loads(batch_path.read_text()); root = Path(batch['controller'])
    sys.path.insert(0, str(root))
    import part3_band as workflow
    from part3_existing_audit import aggregate
    from extension_repeat_selection import digest, read
    import torch
    torch.set_num_threads(1)
    assert batch['status'] == 'completed_band_not_qualified' and not batch['errors']
    workflow.engine.verify(batch)
    out.mkdir(parents=True, exist_ok=False)
    checks = implementation_checks(); print('Implementation checks passed', flush=True)
    children = {seed: json.loads(Path(p).read_text()) for seed, p in batch['children'].items()}
    parent = {(r['condition'], r['example_id']): r for r in read(root/'parent_scores.jsonl')}
    examples = {e['example_id']: e for split in workflow.SPLITS
                for e in read(root/'data'/split/'band.jsonl')}
    compared = []; integrity = []; tokens = []; raw_cases = []
    for seed, child in children.items():
        for split in workflow.SPLITS:
            for condition in ('base', 'adapter', 'band'):
                result = workflow.audit_run(child, condition+'-'+split)
                integrity.append(dict(seed=seed, split=split, condition=condition, passed=result['checks_passed']))
                if condition != 'band': continue
                entry = child['runs']['band-'+split]['adapter']
                raw = {r['example_id']: r for r in read(Path(entry['output'])/'planned_responses.jsonl') if r['phase']=='primary'}
                for r in result['scores']:
                    full = parent[seed, r['example_id']]; e = examples[r['example_id']]
                    changes = transitions(r, full)
                    compared.append(dict(r, seed=seed, **changes,
                        full_answer=full['answer'], assigned_choice_index=e['choices'].index(e['assigned_goal_answer']),
                        recipient_choice_index=e['choices'].index(e['recipient_optimal_answer']),
                        assigned_label=e['assigned_goal_answer'], recipient_label=e['recipient_optimal_answer']))
                    if r['cohort']=='anchor' and r['alignment']=='opposed' and r['query']=='message':
                        raw_cases.append(dict(seed=seed, example_id=r['example_id'], prompt=e['prompt'],
                            full_answer=full['answer'], band_answer=r['answer'],
                            band_raw_response=raw[r['example_id']]['raw_response'], **changes))
                tokens.append(dict(seed=seed, split=split,
                    subset=result['intervention']['subset'], versions=result['manifest']['versions']))
            print(seed, split, 're-audited', flush=True)
    metrics = tuple(transitions(compared[0], parent[compared[0]['seed'], compared[0]['example_id']]))
    summary = aggregate(compared, ('seed','split','cohort','variant','query','alignment'), metrics)
    primary = [r for r in compared if r['cohort']=='anchor' and r['query']=='message' and r['alignment']=='opposed']
    strata = {}
    for fields in [('role',), ('instance_id',), ('table_order',), ('answer_order',),
                   ('assigned_choice_index',), ('recipient_choice_index',), ('role','instance_id'),
                   ('role','table_order','answer_order')]:
        strata['_'.join(fields)] = aggregate(primary, ('seed','split','variant')+fields, metrics)
    joint = defaultdict(list)
    for r in primary:
        if r['variant']=='anchor_reminder': joint[r['example_id']].append(r)
    patterns = []
    for eid, rows in joint.items():
        rows.sort(key=lambda r:r['seed']); assert len(rows)==3
        patterns.append(dict(example_id=eid, split=rows[0]['split'], role=rows[0]['role'],
            band_pattern=''.join('R' if r['recipient_goal'] else 'A' if r['assigned_goal'] else '?' for r in rows),
            full_pattern=''.join('R' if parent[r['seed'],eid]['recipient_goal'] else 'A' for r in rows)))
    weight_rows, cosine, training = weights(children)
    totals = aggregate(weight_rows, ('seed','module','region','retained'), ('norm2','zero'))
    outputs = dict(summary=summary, strata=strata, cross_seed_patterns=patterns,
        implementation_checks=checks, integrity=integrity, runtime=tokens,
        weight_modules=weight_rows, weight_totals=totals, weight_cosine=cosine, training=training)
    for name, data in outputs.items(): workflow.engine.write(out/(name+'.json'), data)
    workflow.engine.jsonl(out/'comparisons.jsonl', compared)
    workflow.engine.jsonl(out/'historical_opposed_messages.jsonl', raw_cases)
    workflow.engine.write(out/'complete.json', dict(passed=True, comparisons=len(compared), audits=len(integrity),
        completed_utc=datetime.now(timezone.utc).isoformat(), parent=str(batch_path), parent_sha256=digest(batch_path),
        script_sha256=digest(__file__), implementation_checks=checks,
        interpretation='Exposed exploratory data; rotations are not independent worlds; weight norms are not causal attribution.'))
    print(json.dumps([r for r in summary if r['cohort']=='anchor' and r['query']=='message' and r['alignment']=='opposed'], indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); main(args.batch, args.out)
