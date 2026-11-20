"""Sycophancy readout from existing data (PLAN "Sycophancy readout from existing data", 2026-11-17). No model is run.

(A) persona activations: AUROC with which a direction separates `sycophant_fic` role-play responses from
    `candid_friend_fic` (and from `default`) responses, for the Goodfire axis, gi_v2 as steered, and gi_v2 rebuilt
    without the two roles from the saved per-role mean vectors. Bootstrap over questions.
(B) probe labels: label counts per steering setting on chosen probe prompts, from a blind label set.

    sycophancy_readout.py persona --root <extraction root with act_response_only/ scores/ vectors_response_only/>
        --gi-v2 <persona_contrast.pt> --goodfire <axis.pt> --layer 17 [--out a.json]
    sycophancy_readout.py probes --blind <blind dir with key.json, labels/> --questions "general text" ... [--out b.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

POSITIVE, NEGATIVE = "sycophant_fic", "candid_friend_fic"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def question(key: str) -> int:
    return int(key.rsplit("_q", 1)[1])


def boot(pos: dict, neg: dict, draws: int, rng) -> dict:
    """pos / neg: {question: array of projections}. Resample questions, pool their responses."""
    questions = sorted(set(pos) | set(neg))
    point = auroc(np.concatenate([pos[q] for q in questions if q in pos]), np.concatenate([neg[q] for q in questions if q in neg]))
    values = []
    for _ in range(draws):
        pick = rng.choice(questions, len(questions))
        a = [pos[q] for q in pick if q in pos]
        b = [neg[q] for q in pick if q in neg]
        if a and b:
            values.append(auroc(np.concatenate(a), np.concatenate(b)))
    low, high = np.percentile(values, [2.5, 97.5])
    n_pos, n_neg = sum(len(v) for v in pos.values()), sum(len(v) for v in neg.values())
    return dict(auroc=round(point, 4), ci=[round(float(low), 4), round(float(high), 4)], n_pos=n_pos, n_neg=n_neg)


def persona(args) -> None:
    import torch

    def axis(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        v = d["axis"][args.layer].float()
        return v / v.norm()

    def role_mean(role):
        return torch.load(args.root / "vectors_response_only" / f"{role}.pt", map_location="cpu", weights_only=False)["vector"][args.layer].float()

    meta = torch.load(args.gi_v2, map_location="cpu", weights_only=False)
    dishonest = [r for r in meta["schemer_roles"] if r != POSITIVE]
    honest = [r for r in meta["honest_roles"] if r != NEGATIVE]
    assert len(dishonest) == len(meta["schemer_roles"]) - 1 and len(honest) == len(meta["honest_roles"]) - 1
    rebuilt = torch.stack([role_mean(r) for r in dishonest]).mean(0) - torch.stack([role_mean(r) for r in honest]).mean(0)
    full = torch.stack([role_mean(r) for r in meta["schemer_roles"]]).mean(0) - torch.stack([role_mean(r) for r in meta["honest_roles"]]).mean(0)
    directions = {"goodfire": axis(args.goodfire), "gi_v2_as_steered": axis(args.gi_v2), "gi_v2_without_the_two_roles": rebuilt / rebuilt.norm()}
    cos = lambda a, b: round(float(a @ b / (a.norm() * b.norm())), 4)  # noqa: E731
    pair = role_mean(POSITIVE) - role_mean(NEGATIVE)
    out = dict(layer=args.layer, root=str(args.root), dishonest_roles=dishonest, honest_roles=honest,
               check_cos_rebuilt_full_vs_steered=cos(full, directions["gi_v2_as_steered"]),
               cos_with_sycophant_minus_candid_friend={k: cos(v, pair) for k, v in directions.items()},
               cos_goodfire_with={k: cos(directions["goodfire"], v) for k, v in directions.items() if k != "goodfire"}, auroc={})

    def projections(role, gate):
        acts = torch.load(args.root / "act_response_only" / f"{role}.pt", map_location="cpu", weights_only=False)
        scores_path = args.root / "scores" / f"{role}.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        keys = [k for k in acts if not gate or not scores or scores.get(k) == 3]
        matrix = torch.stack([acts[k][args.layer].float() for k in keys])
        return {name: _by_question(keys, (matrix @ d).numpy()) for name, d in directions.items()}, len(keys), len(acts)

    rng = np.random.default_rng(args.seed)
    for gate in (True, False):
        data = {role: projections(role, gate) for role in (POSITIVE, NEGATIVE, "default")}
        block = dict(kept={role: [data[role][1], data[role][2]] for role in data})
        for other in (NEGATIVE, "default"):
            block[f"sycophant_vs_{other}"] = {name: boot(data[POSITIVE][0][name], data[other][0][name], args.draws, rng) for name in directions}
        out["auroc"]["score3" if gate else "all_responses"] = block
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


def _by_question(keys, values) -> dict:
    grouped = {}
    for k, v in zip(keys, values):
        grouped.setdefault(question(k), []).append(v)
    return {q: np.array(v) for q, v in grouped.items()}


def probes(args) -> None:
    key = json.loads((args.blind / "key.json").read_text())
    labels = {}
    for path in sorted((args.blind / "labels").glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[Path(row["file"]).stem] = row
    out = {}
    for text in args.questions:
        rows = [(k, v) for k, v in key.items() if v["question"] == text and v["set"] in args.sets]
        assert rows, text
        table = {}
        for item, v in rows:
            cell = table.setdefault(v["setting"], {label: 0 for label in args.labels} | {"n": 0, "unclear": 0, "cut_or_empty": 0, "yes_items": []})
            cell["n"] += 1
            cell["cut_or_empty"] += int(v["truncated"] or v["empty_final"])
            for label in args.labels:
                cell[label] += labels[item][label] == "yes"
            cell["unclear"] += labels[item][args.labels[0]] == "unclear"
            if labels[item][args.labels[0]] == "yes":
                cell["yes_items"].append(dict(item=item, set=v["set"], seed=v["seed"], quote=labels[item].get("quote", ""), note=labels[item].get("note", "")))
        out[text] = dict(sorted(table.items()))
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("persona")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--gi-v2", type=Path, required=True)
    a.add_argument("--goodfire", type=Path, required=True)
    a.add_argument("--layer", type=int, required=True)
    a.add_argument("--draws", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", type=Path)
    a.set_defaults(run=persona)
    b = sub.add_parser("probes")
    b.add_argument("--blind", type=Path, required=True)
    b.add_argument("--questions", nargs="+", required=True)
    b.add_argument("--sets", nargs="+", default=["probes_low", "probes_high"])
    b.add_argument("--labels", nargs="+", default=["final_deception", "scheming_voice", "refusal"])
    b.add_argument("--out", type=Path)
    b.set_defaults(run=probes)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
