"""Size-match a cap on one direction to a reference cap on another (PLAN "gpt-oss random-direction p50 cap probe", size amendment
2026-11-10).

Two caps at the same percentile need not move the residual by the same amount: the per-token projection is spread much more
widely along some directions than along others. The open-loop size of a cap at layer l is the mean over unsteered turns of
E[max(projection - tau_l, 0)], the mean amount by which the cap lowers the projection (in projection units = residual units,
the directions are applied as unit vectors). It is read from a cap_thresholds.sbatch projection sidecar, which stores per-turn
quantiles of the generated-token projection (every 5th percentile): each stored quantile stands for an equal share of tokens,
so mass beyond p5 / p95 is not seen (the same approximation on both directions).

For every layer this script finds the threshold tau'_l on the target direction whose open-loop size equals the reference cap's
and writes {layer: tau'} in the --cap-thresholds format. It also prints, per layer, the size of the target's own same-percentile
cap relative to the reference (the "gentler by" factor) and the share of target tokens above tau'.

Usage: cap_size_match.py --ref-sidecar <proj.jsonl> --ref-tau <{layer: tau}.json> --target-sidecar <proj.jsonl> --out <json>
       [--target-quantile 0.5]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def turn_quantiles(sidecar: Path, layer: int) -> list:
    """Per turn with generated tokens: (sorted stored quantile values, {quantile key: value})."""
    out = []
    for line in sidecar.open():
        g = json.loads(line)["layers"].get(str(layer), {}).get("generated_projection")
        if g and g.get("n", 0) > 0:
            out.append((np.array(sorted(g["quantiles"].values()), dtype=float), g["quantiles"]))
    assert out, f"no turn with generated tokens at layer {layer} in {sidecar}"
    return out


def cap_size(turns: list, tau: float) -> float:
    """Open-loop size: mean over turns of the mean over stored quantiles of max(q - tau, 0)."""
    return float(np.mean([np.clip(v - tau, 0, None).mean() for v, _ in turns]))


def share_above(turns: list, tau: float) -> float:
    return float(np.mean([(v > tau).mean() for v, _ in turns]))


def match_tau(turns: list, size: float) -> float:
    """tau with cap_size(turns, tau) == size; cap_size falls monotonically in tau and is linear below every stored quantile."""
    assert size > 0
    hi = max(v.max() for v, _ in turns)           # size 0
    lo = min(v.min() for v, _ in turns) - size - 1  # every token is above: size >= size + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if cap_size(turns, mid) > size:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    assert abs(cap_size(turns, tau) - size) <= 1e-6 * max(1.0, size), (cap_size(turns, tau), size)
    return tau


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-sidecar", type=Path, required=True)
    ap.add_argument("--ref-tau", type=Path, required=True, help="thresholds of the reference cap, {layer: tau}")
    ap.add_argument("--target-sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="size-matched thresholds for the target direction")
    ap.add_argument("--target-quantile", default="0.5", help="stored quantile of the target's own same-percentile cap, for the comparison column")
    args = ap.parse_args()
    ref_tau = {int(k): float(v) for k, v in json.loads(args.ref_tau.read_text()).items()}
    out, ratios = {}, []
    print(f"ref {args.ref_sidecar} @ {args.ref_tau.name}\ntarget {args.target_sidecar}")
    print("layer   ref_tau  ref_size | target_p_tau  target_p_size  size/ref | matched_tau  share_above  turns(ref/target)")
    for layer in sorted(ref_tau):
        ref, tgt = turn_quantiles(args.ref_sidecar, layer), turn_quantiles(args.target_sidecar, layer)
        size = cap_size(ref, ref_tau[layer])
        own_tau = float(np.median([q[args.target_quantile] for _, q in tgt]))  # as cap_thresholds.sbatch: median over turns
        own = cap_size(tgt, own_tau)
        tau = match_tau(tgt, size)
        out[str(layer)] = tau
        ratios.append(own / size)
        print(f"{layer:>5} {ref_tau[layer]:>9.1f} {size:>9.1f} | {own_tau:>12.1f} {own:>14.1f} {own / size:>9.3f} | {tau:>11.1f} {share_above(tgt, tau):>12.2f}"
              f"  {len(ref)}/{len(tgt)}")
    print(f"same-percentile target cap, size relative to the reference: band mean {np.mean(ratios):.3f} (min {min(ratios):.3f}, max {max(ratios):.3f})")
    tmp = args.out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
