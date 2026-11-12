"""Cap thresholds at a percentile the projection sidecar did not store, interpolated from the stored ones.

cap_thresholds.sbatch writes tau at p25 and p50 from the per-turn quantiles of the generated-token projection
(src/selfconcept/codebench/hf_harmony.py QUANTILES). For a percentile between two stored ones (p35 between p25 and
p50, say) this script interpolates per turn on the normal-score scale -- the per-turn value sits where a Gaussian
with the two stored quantiles would put it (w = (z_q - z_lo) / (z_hi - z_lo), 0.429 for p35 between p25 and p50;
plain linear would be 0.4) -- and then takes the median over turns, exactly as the sbatch does for the stored
percentiles. The per-turn interpolation is monotone in q, so the ladder p25 < p35 < p50 holds layer by layer on the
same episodes; a fresh thresholds run would sample different episodes and could cross. Writes {layer: tau} in the
format steered_rh_vllm.py --cap-thresholds reads plus a <out>.note.json with the method and the per-layer stored
neighbours.

    python scripts/cap_threshold_interp.py <proj_*_projection.jsonl> 0.35 --out <dir>/<name>_p35.json [--layers 20 ... 38] [--linear]
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def interpolate(rows: list[dict], q: float, layers: list[int] | None, linear: bool) -> tuple[dict, dict]:
    stored = sorted(float(k) for k in rows[0]["layers"][next(iter(rows[0]["layers"]))]["generated_projection"]["quantiles"])
    if str(q) in {str(s) for s in stored}:
        lo = hi = q
    else:
        below = [s for s in stored if s < q]
        above = [s for s in stored if s > q]
        if not below or not above:
            raise SystemExit(f"{q} is outside the stored quantiles {stored}")
        lo, hi = below[-1], above[0]
    if lo == hi:
        w = 0.0
    elif linear:
        w = (q - lo) / (hi - lo)
    else:
        z = statistics.NormalDist().inv_cdf
        w = (z(q) - z(lo)) / (z(hi) - z(lo))
    present = sorted({int(k) for r in rows for k in r["layers"]})
    layers = present if layers is None else [l for l in layers if l in present]
    tau, note = {}, {}
    for layer in layers:
        per_turn = [r["layers"][str(layer)]["generated_projection"] for r in rows
                    if str(layer) in r["layers"] and r["layers"][str(layer)]["generated_projection"].get("n", 0) > 0]
        v_lo = np.array([t["quantiles"][str(lo)] for t in per_turn])
        v_hi = np.array([t["quantiles"][str(hi)] for t in per_turn])
        values = v_lo + w * (v_hi - v_lo)
        tau[str(layer)] = float(np.median(values))
        note[str(layer)] = {"turns": len(per_turn), f"median_p{round(lo * 100)}": float(np.median(v_lo)),
                            f"median_p{round(hi * 100)}": float(np.median(v_hi))}
    method = {"quantile": q, "stored_neighbours": [lo, hi], "weight": w,
              "scale": "linear" if linear else "normal-score", "turns": len(rows)}
    return tau, {"method": method, "layers": note}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("projection", type=Path, help="proj_<scenario>_projection.jsonl of the thresholds run")
    parser.add_argument("quantile", type=float, help="target percentile as a fraction, e.g. 0.35")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="default: every layer in the sidecar")
    parser.add_argument("--linear", action="store_true", help="interpolate linearly in q instead of on normal scores")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.projection.open()]
    tau, note = interpolate(rows, args.quantile, args.layers, args.linear)
    note["source"] = str(args.projection)
    args.out.write_text(json.dumps(tau, indent=1))
    args.out.with_suffix(".note.json").write_text(json.dumps(note, indent=1))
    m = note["method"]
    print(f"p{round(args.quantile * 100)} from p{round(m['stored_neighbours'][0] * 100)} / "
          f"p{round(m['stored_neighbours'][1] * 100)}, weight {m['weight']:.3f} ({m['scale']}), {m['turns']} turns")
    for layer, v in tau.items():
        n = note["layers"][layer]
        keys = [k for k in n if k.startswith("median_p")]
        if len(keys) == 2:
            lo_k, hi_k = keys
            print(f"  L{layer}: {n[lo_k]:8.1f} < {v:8.1f} < {n[hi_k]:8.1f}   ({lo_k[7:]} / p{round(args.quantile * 100)} / {hi_k[7:]})")
        else:
            print(f"  L{layer}: {v:8.1f}   (stored {keys[0][7:]}, {n['turns']} turns)")
    print("written", args.out)


if __name__ == "__main__":
    main()
