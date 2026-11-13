"""Two directions on one model's unsteered episodes: gi_v2 against the Goodfire reward-hacking directions, from the
cache of axis_acts_extract.py. Per layer:

- cosine of each Goodfire readout's direction with gi_v2, with the per-task sign-flip null of
  goodfire_rh_geometry.py when its JSON is given (--geometry-json) and the isotropic sd 1/sqrt(hidden);
- episode level, reasoning and answer segments apart: Pearson and Spearman correlation of the two projections of the
  episode's token-mean residual; the same within problems (both projections demeaned per problem) and between
  problems (problem means); a problem-cluster bootstrap interval; and the null that matters here, the correlation of
  each direction's projection with the projections onto random unit directions (--random, drawn here and applied to
  the stored means). Residual-stream episode means share a few dominant components, so two unrelated directions
  correlate far from zero; an observed r is read against that band (percentile among the random directions), not
  against zero. As a sensitivity the top 1 and top 5 principal components of the episode means are removed first;
- a second, harder episode-level null when --control-dirs is given: a folder of real contrast directions of the same
  model (one (layers, hidden) "axis" per .pt, e.g. the isolated antonym controls). Isotropic random directions carry
  almost none of the dominant components, real difference-of-means directions do, so the question "is this r more
  than any two activation-derived directions show" is answered against this population: the percentile of the
  observed |r| among |r(gi_v2, control)| and among |r(Goodfire, control)|, raw, within problems and with the top
  components removed, plus the |r| distribution over control pairs;
- token level: the correlation of the two projections over all tokens (pooled, and within episodes), from the token
  sums the extraction stored, against the same statistic for the stored random directions.

    python scripts/axis_acts_analysis.py --acts <dir with unsteered__seed*.pt> --label gpt-oss-120b --steer-layer 17
        --band 12 23 --geometry-json <goodfire_rh_geometry.json> [<report_split json>] --out <axis_compare.json>

Writes the JSON and prints the steer-layer and band rows. Chart: axis_acts_chart.py. CPU.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

_spec = importlib.util.spec_from_file_location("hack_lens_extract", Path(__file__).with_name("hack_lens_extract.py"))
hack_lens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hack_lens)

SEGMENTS = ("reasoning", "answer")
REFERENCE = "gi_v2"
GEOMETRY_READOUT = {"gf_all": "all_response", "gf_close": "close", "gf_last": "last", "gf_tool": "tool_calls", "gf_report": "report"}
PC_REMOVED = (1, 5)
N_BOOTSTRAP = 1000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--arm", default="unsteered")
    ap.add_argument("--label", required=True)
    ap.add_argument("--steer-layer", type=int, required=True)
    ap.add_argument("--band", type=int, nargs=2, required=True, help="first and last layer of the gi_v2 cap band")
    ap.add_argument("--geometry-json", type=Path, nargs="*", default=[], help="goodfire_rh_geometry.json files (sign-flip nulls of the cosine)")
    ap.add_argument("--random", type=int, default=1000, help="random unit directions for the episode-level null")
    ap.add_argument("--control-dirs", type=Path, default=None, help="folder of control direction .pt files (dict with 'axis' (layers, hidden)): activation-derived null")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def columns_r(x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Pearson r of x (E,) with every column of columns (E, R)."""
    xc = x - x.mean()
    cc = columns - columns.mean(0)
    return (xc @ cc) / np.maximum(np.linalg.norm(xc) * np.linalg.norm(cc, axis=0), 1e-30)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(columns_r(x, y[:, None])[0])


def load_cache(acts: Path, arm: str) -> dict:
    files = sorted(acts.glob(f"{arm}__seed*.pt"), key=lambda p: int(p.stem.split("seed")[1]))
    if not files:
        raise SystemExit(f"no {arm}__seed*.pt in {acts}")
    parts = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    first = parts[0]
    cache = {key: first[key] for key in ("model", "family", "record_layers", "directions", "named_directions", "direction_specs", "chat_kwargs", "max_length")}
    cache["seeds"] = [part["seed"] for part in parts]
    cache["example_id"] = [example for part in parts for example in part["example_id"]]
    cache["seed_of_episode"] = [part["seed"] for part in parts for _ in part["example_id"]]
    cache["truncated_turns"] = sum(len(t) for part in parts for t in part["truncated_turns"])
    cache["oom_truncated"] = sum(len(t) for part in parts for t in part["oom_truncated"])
    cache["missing_reasoning"] = sum(len(t) for part in parts for t in part["missing_reasoning"])
    for segment in SEGMENTS:
        cache[f"{segment}_mean"] = torch.cat([part[f"{segment}_mean"] for part in parts])
        cache[segment] = {key: np.concatenate([part["stats"][segment][key] for part in parts]) for key in first["stats"][segment]}
    return cache


def group_matrix(example_ids: list[str]) -> np.ndarray:
    """(E, problems) one-hot of each episode's problem."""
    problems = sorted(set(example_ids))
    index = {problem: i for i, problem in enumerate(problems)}
    onehot = np.zeros((len(example_ids), len(problems)))
    onehot[np.arange(len(example_ids)), [index[example] for example in example_ids]] = 1.0
    return onehot


def demean(values: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """values (E,) or (E, R) minus the mean of the episode's problem."""
    counts = onehot.sum(0)
    return values - onehot @ ((onehot.T @ values) / (counts[:, None] if values.ndim == 2 else counts))


def problem_means(values: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    counts = onehot.sum(0)
    return (onehot.T @ values) / (counts[:, None] if values.ndim == 2 else counts)


def null_summary(observed: float, null: np.ndarray) -> dict:
    absolute = np.abs(null)
    return {"null_abs_median": float(np.median(absolute)), "null_abs_95": float(np.percentile(absolute, 95)),
            "percentile": float((absolute < abs(observed)).mean())}


def bootstrap_interval(x: np.ndarray, y: np.ndarray, onehot: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Percentile interval of Pearson r, problems resampled with replacement (episodes of a problem stay together)."""
    weights = rng.multinomial(onehot.shape[1], np.full(onehot.shape[1], 1.0 / onehot.shape[1]), size=N_BOOTSTRAP) @ onehot.T  # (B, E)
    total = weights.sum(1)
    mean_x, mean_y = weights @ x / total, weights @ y / total
    covariance = weights @ (x * y) / total - mean_x * mean_y
    variance_x, variance_y = weights @ (x * x) / total - mean_x ** 2, weights @ (y * y) / total - mean_y ** 2
    r = covariance / np.sqrt(np.maximum(variance_x * variance_y, 1e-30))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def episode_level(x: np.ndarray, y: np.ndarray, null: np.ndarray, onehot: np.ndarray, rng: np.random.Generator) -> dict:
    """x, y (E,) projections of the two named directions; null (E, R) projections onto random directions."""
    out = {"r": pearson(x, y), "spearman": pearson(rankdata(x), rankdata(y)), "ci95": bootstrap_interval(x, y, onehot, rng)}
    out["vs_random"] = {"reference": null_summary(out["r"], columns_r(x, null)), "goodfire": null_summary(out["r"], columns_r(y, null))}
    xw, yw, nullw = demean(x, onehot), demean(y, onehot), demean(null, onehot)
    out["within_problem"] = {"r": pearson(xw, yw), "vs_random": {"reference": null_summary(pearson(xw, yw), columns_r(xw, nullw)),
                                                               "goodfire": null_summary(pearson(xw, yw), columns_r(yw, nullw))}}
    xb, yb, nullb = problem_means(x, onehot), problem_means(y, onehot), problem_means(null, onehot)
    out["between_problem"] = {"r": pearson(xb, yb), "vs_random": {"reference": null_summary(pearson(xb, yb), columns_r(xb, nullb)),
                                                                "goodfire": null_summary(pearson(xb, yb), columns_r(yb, nullb))}}
    return out


def control_level(x: np.ndarray, y: np.ndarray, control: np.ndarray, onehot: np.ndarray) -> dict:
    """Observed r(x, y) against the control population: control (E, C) projections onto real contrast directions."""
    def block(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict:
        r = pearson(a, b)
        return {"r": r, "reference": null_summary(r, columns_r(a, c)), "goodfire": null_summary(r, columns_r(b, c))}
    return {"episode": block(x, y, control), "within_problem": block(demean(x, onehot), demean(y, onehot), demean(control, onehot))}


def control_pairs(control: np.ndarray) -> dict:
    """|r| over all pairs of control directions: how correlated two unrelated real contrast directions are here."""
    absolute = np.abs(np.corrcoef(control.T)[np.triu_indices(control.shape[1], 1)])
    return {"abs_median": float(np.median(absolute)), "abs_95": float(np.percentile(absolute, 95)), "abs_max": float(absolute.max())}


def token_level(stats: dict, layer_index: int, a: int, b: int, named: int) -> dict:
    """Token-level correlation of directions a and b (indices into the named directions) from the stored sums, pooled
    over all tokens and within episodes, with the same statistic of a and of b against every stored random direction."""
    n = stats["n"].astype(np.float64)                     # (E,)
    mean = stats["proj"][:, layer_index]                  # (E, K), nan where the episode has no token of the segment
    square, cross = stats["proj_square_sum"][:, layer_index], stats["proj_cross_sum"][:, layer_index]  # (E, K), (E, named, K)
    valid = n > 0
    n, mean, square, cross = n[valid], mean[valid], square[valid], cross[valid]
    total = n.sum()
    grand = (n[:, None] * mean).sum(0) / total
    pooled_variance = square.sum(0) / total - grand ** 2
    within_variance = (square - n[:, None] * mean ** 2).sum(0)

    def r_with(i: int, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled = (cross[:, i, columns].sum(0) / total - grand[i] * grand[columns]) / np.sqrt(np.maximum(pooled_variance[i] * pooled_variance[columns], 1e-30))
        within = (cross[:, i, columns] - n[:, None] * mean[:, [i]] * mean[:, columns]).sum(0) / np.sqrt(np.maximum(within_variance[i] * within_variance[columns], 1e-30))
        return pooled, within

    nulls = np.arange(named, mean.shape[1])
    pooled, within = (float(v[0]) for v in r_with(a, np.array([b])))
    null_a, null_b = r_with(a, nulls), r_with(b, nulls)
    return {"tokens": int(total), "pooled": {"r": pooled, "vs_random": {"reference": null_summary(pooled, null_a[0]), "goodfire": null_summary(pooled, null_b[0])}},
            "within_episode": {"r": within, "vs_random": {"reference": null_summary(within, null_a[1]), "goodfire": null_summary(within, null_b[1])}}}


def main() -> None:
    args = parse_args()
    cache = load_cache(args.acts, args.arm)
    layers, names, named = cache["record_layers"], cache["directions"], cache["named_directions"]
    goodfire_names = [name for name in named if name != REFERENCE]
    num_layers, hidden = hack_lens.model_dims(cache["model"])
    directions = dict(hack_lens.load_direction_matrix(spec, num_layers, hidden) for spec in cache["direction_specs"])  # unit rows
    geometry: dict[str, dict] = {}
    for path in args.geometry_json:
        for readout, contrasts in json.loads(path.read_text())["by_readout"].items():
            geometry.setdefault(readout, contrasts["pl_all_families"])
    controls = None
    if args.control_dirs is not None:
        files = sorted(args.control_dirs.glob("*.pt"))
        stack = torch.stack([torch.load(path, map_location="cpu", weights_only=False)["axis"].float() for path in files])  # (C, layers, hidden)
        if stack.shape[1:] != (num_layers, hidden):
            raise SystemExit(f"{args.control_dirs}: control shape {tuple(stack.shape[1:])}, expected {(num_layers, hidden)}")
        controls = stack / stack.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    onehot = group_matrix(cache["example_id"])
    rng = np.random.default_rng(20261109)
    result = {"label": args.label, "model": cache["model"], "acts": str(args.acts), "seeds": cache["seeds"], "episodes": len(cache["example_id"]),
              "problems": int(onehot.shape[1]), "layers": layers, "steer_layer": args.steer_layer, "band": args.band, "hidden": hidden,
              "isotropic_sd": hidden ** -0.5, "chat_kwargs": cache["chat_kwargs"], "random_directions": args.random,
              "stored_null_directions": len(names) - len(named), "truncated_turns": cache["truncated_turns"], "oom_truncated": cache["oom_truncated"],
              "missing_reasoning_turns": cache["missing_reasoning"], "direction_specs": cache["direction_specs"],
              "episodes_with_tokens": {s: int((cache[s]["n"] > 0).sum()) for s in SEGMENTS}, "tokens": {s: int(cache[s]["n"].sum()) for s in SEGMENTS},
              "cosine": {}, "episode": {s: {g: [] for g in goodfire_names} for s in SEGMENTS}, "token": {s: {g: [] for g in goodfire_names} for s in SEGMENTS},
              "pc1_variance_share": {s: [] for s in SEGMENTS}, "projection_check": {s: [] for s in SEGMENTS},
              "control_dirs": str(args.control_dirs) if controls is not None else None, "control_directions": 0 if controls is None else int(controls.shape[0]),
              "control_pairs": {s: [] for s in SEGMENTS}, "control_cosine": {}}
    if controls is not None:  # |cos| of each named direction with the controls, per recorded layer: the cosine-level counterpart of the null
        for name in named:
            cos = torch.stack([controls[:, layer] @ directions[name][layer] for layer in layers]).abs()  # (L, C)
            result["control_cosine"][name] = {"abs_median": cos.median(1).values.tolist(), "abs_95": torch.quantile(cos, 0.95, dim=1).tolist()}
    for name in goodfire_names:
        entry = {"cosine": [float(directions[name][layer] @ directions[REFERENCE][layer]) for layer in layers]}
        readout = geometry.get(GEOMETRY_READOUT.get(name, ""))
        if readout is not None:
            entry["sign_flip_null_abs_95"] = [readout["sign_flip_null_abs_cosine_95th"][REFERENCE][layer] for layer in layers]
            entry["split_half"] = [readout["split_half_over_tasks_mean"][layer] for layer in layers]
            entry["validation_auroc"] = [readout["validation_auroc"][layer] for layer in layers]
        result["cosine"][name] = entry

    reference_index = named.index(REFERENCE)
    for segment in SEGMENTS:
        stats = cache[segment]
        valid = stats["n"] > 0
        onehot_valid = onehot[valid][:, onehot[valid].sum(0) > 0]
        for li, layer in enumerate(layers):
            means = cache[f"{segment}_mean"][valid, li].float()                                     # (E, hidden)
            generator = torch.Generator().manual_seed(1_000_003 * layer + 17)
            random_units = torch.nn.functional.normalize(torch.randn(args.random, hidden, generator=generator), dim=-1)
            null = (means @ random_units.T).double().numpy()                                          # (E, R)
            exact = stats["proj"][valid, li]                                                          # (E, K) from the float64 token sums
            named_units = torch.stack([directions[name][layer] for name in named])                    # (named, hidden)
            offline = (means @ named_units.T).double().numpy()
            result["projection_check"][segment].append(float(np.abs(offline - exact[:, :len(named)]).max() / max(np.abs(exact[:, :len(named)]).max(), 1e-30)))
            centered = means - means.mean(0)
            torch.manual_seed(1_000_003 * layer + 29)                                                 # svd_lowrank draws its test matrix from the global generator
            _, singular, components = torch.svd_lowrank(centered, q=max(PC_REMOVED) + 5, niter=4)     # components (hidden, q)
            result["pc1_variance_share"][segment].append(float(singular[0] ** 2 / centered.pow(2).sum()))
            scores = (centered @ components).double().numpy()                                         # (E, q)
            loading_named = (components.T @ named_units.T).double().numpy()                           # (q, named)
            loading_null = (components.T @ random_units.T).double().numpy()                           # (q, R)
            x = exact[:, reference_index]
            if controls is not None:
                control_units = controls[:, layer]                                                    # (C, hidden)
                control = (means @ control_units.T).double().numpy()                                  # (E, C)
                loading_control = (components.T @ control_units.T).double().numpy()                   # (q, C)
                result["control_pairs"][segment].append(control_pairs(control))
            for name in goodfire_names:
                gi = named.index(name)
                entry = episode_level(x, exact[:, gi], null, onehot_valid, rng)
                for k in PC_REMOVED:
                    xk = offline[:, reference_index] - scores[:, :k] @ loading_named[:k, reference_index]
                    yk = offline[:, gi] - scores[:, :k] @ loading_named[:k, gi]
                    nullk = null - scores[:, :k] @ loading_null[:k]
                    rk = pearson(xk, yk)
                    entry[f"top{k}_pc_removed"] = {"r": rk, "vs_random": {"reference": null_summary(rk, columns_r(xk, nullk)), "goodfire": null_summary(rk, columns_r(yk, nullk))}}
                if controls is not None:
                    entry["vs_control"] = control_level(x, exact[:, gi], control, onehot_valid)
                    for k in PC_REMOVED:
                        xk = offline[:, reference_index] - scores[:, :k] @ loading_named[:k, reference_index]
                        yk = offline[:, gi] - scores[:, :k] @ loading_named[:k, gi]
                        controlk = control - scores[:, :k] @ loading_control[:k]
                        entry["vs_control"][f"top{k}_pc_removed"] = {"reference": null_summary(pearson(xk, yk), columns_r(xk, controlk)),
                                                                      "goodfire": null_summary(pearson(xk, yk), columns_r(yk, controlk))}
                result["episode"][segment][name].append(entry)
                result["token"][segment][name].append(token_level(stats, li, reference_index, gi, len(named)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    band = [li for li, layer in enumerate(layers) if args.band[0] <= layer <= args.band[1]]
    steer = layers.index(args.steer_layer)
    print(f"{args.label}: {result['episodes']} episodes, {result['problems']} problems, seeds {cache['seeds']}; truncated turns {cache['truncated_turns']}, "
          f"out-of-memory retries {cache['oom_truncated']}; projection check (fp16 means vs exact) max {max(max(v) for v in result['projection_check'].values()):.2e}")
    print(f"| direction | cos L{args.steer_layer} (null95) | band cos | segment | episode r L{args.steer_layer} [CI] | random-null |r| 95 (gi / gf) | pct (gi / gf) | within-problem r (pct gi / gf) | top-1-PC-removed r (pct) | token r pooled (pct) | token r within episode (pct) | band mean episode r |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in goodfire_names:
        cosine = result["cosine"][name]
        for segment in SEGMENTS:
            e, t = result["episode"][segment][name][steer], result["token"][segment][name][steer]
            null95 = f" ({cosine['sign_flip_null_abs_95'][steer]:.3f})" if "sign_flip_null_abs_95" in cosine else ""
            print(f"| {name} | {cosine['cosine'][steer]:+.3f}{null95} | {np.mean([cosine['cosine'][i] for i in band]):+.3f} | {segment} | "
                  f"{e['r']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] | {e['vs_random']['reference']['null_abs_95']:.2f} / {e['vs_random']['goodfire']['null_abs_95']:.2f} | "
                  f"{e['vs_random']['reference']['percentile']:.2f} / {e['vs_random']['goodfire']['percentile']:.2f} | "
                  f"{e['within_problem']['r']:+.2f} ({e['within_problem']['vs_random']['reference']['percentile']:.2f} / {e['within_problem']['vs_random']['goodfire']['percentile']:.2f}) | "
                  f"{e['top1_pc_removed']['r']:+.2f} ({e['top1_pc_removed']['vs_random']['reference']['percentile']:.2f} / {e['top1_pc_removed']['vs_random']['goodfire']['percentile']:.2f}) | "
                  f"{t['pooled']['r']:+.2f} ({t['pooled']['vs_random']['reference']['percentile']:.2f} / {t['pooled']['vs_random']['goodfire']['percentile']:.2f}) | "
                  f"{t['within_episode']['r']:+.2f} ({t['within_episode']['vs_random']['reference']['percentile']:.2f} / {t['within_episode']['vs_random']['goodfire']['percentile']:.2f}) | "
                  f"{np.mean([result['episode'][segment][name][i]['r'] for i in band]):+.2f} |")
    if controls is not None:
        print(f"\ncontrol null: {controls.shape[0]} directions from {args.control_dirs}")
        print(f"| direction | segment | episode r L{args.steer_layer} | control |r| median / 95 (vs gi_v2) | pct | control |r| median / 95 (vs Goodfire) | pct | within-problem pct (gi / gf) | top-1-PC pct (gi / gf) | top-5-PC pct (gi / gf) | control-pair |r| median / 95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name in goodfire_names:
            for segment in SEGMENTS:
                c, pairs = result["episode"][segment][name][steer]["vs_control"], result["control_pairs"][segment][steer]
                e, w = c["episode"], c["within_problem"]
                print(f"| {name} | {segment} | {e['r']:+.2f} | {e['reference']['null_abs_median']:.2f} / {e['reference']['null_abs_95']:.2f} | {e['reference']['percentile']:.2f} | "
                      f"{e['goodfire']['null_abs_median']:.2f} / {e['goodfire']['null_abs_95']:.2f} | {e['goodfire']['percentile']:.2f} | "
                      f"{w['reference']['percentile']:.2f} / {w['goodfire']['percentile']:.2f} | "
                      f"{c['top1_pc_removed']['reference']['percentile']:.2f} / {c['top1_pc_removed']['goodfire']['percentile']:.2f} | "
                      f"{c['top5_pc_removed']['reference']['percentile']:.2f} / {c['top5_pc_removed']['goodfire']['percentile']:.2f} | {pairs['abs_median']:.2f} / {pairs['abs_95']:.2f} |")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
