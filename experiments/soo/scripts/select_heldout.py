"""Select, on the development eval sets only, the real and the random all-position offset each model carries into the
held-out round (HELDOUT_ROUND.md). One rule for both.

Usage: python select_heldout.py OUT_DIR
Reads every development snapshot listed in DEV below, reads each all-position offset cell (real mean offsets and
matched-norm random draws, every seed) with the reproduction rule against its own snapshot's same-day reference, and
picks, separately for real and random cells, the minimum of
  (class, max over orientations |deception - reference deception|, alpha, seed)
where class 0 = reproduces (no damage, not budget-bound), 1 = no damage and not budget-bound, 2 = anything else.
A cell run in several snapshots is read from the earliest. Writes OUT_DIR/selection.json and OUT_DIR/selection.md.
No held-out data is read.
"""

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_kimi_loc import loc_verdict  # noqa: E402
from analyze_layer_round import ORIENTS, load, verdict  # noqa: E402

STUDY2 = Path(__file__).resolve().parents[1] / "results/study2"
# Development snapshots per model, oldest first (the partial first Mistral dose snapshot, 20260929T221309Z, is excluded:
# its job was replaced by 20260930T014849Z, which reran every cell).
DEV = {
    "mistral-7b": ["steer-mistral-7b-L16-20260928T230233Z", "dose-mistral-7b-L16-20260930T014849Z", "ctrl-mistral-7b-L16-20261027T182527Z"],
    "gemma4-12b": ["steer-gemma4-12b-L24-20260928T230232Z", "dose-gemma4-12b-L24-20260929T221158Z", "ctrl-gemma4-12b-L24-20261027T182202Z"],
    "gemma4-31b": ["steer-gemma4-31b-L32-20260929T180338Z", "dose-gemma4-31b-L32-20260929T221420Z", "ctrl-gemma4-31b-L32-20261027T182019Z"],
    "qwen38-27b": ["steer-qwen38-27b-L32-20260929T221532Z", "ctrl-qwen38-27b-L32-20261027T182331Z"],
    "kimi-dev-72b": ["kimijoint-kimi-dev-72b-L56-20261001T202619Z", "ctrl-kimi-dev-72b-L56-20261027T182658Z"],
}
# The reference each model's offsets are read against (CONTROL_ROUND.md): the band they were extracted from; on Qwen the
# full adapter (its fused-module band is itself partial); on Kimi the LoRA subset the offset replaces (per cell).
REFERENCE = {"mistral-7b": "band-v", "gemma4-12b": "band-v", "gemma4-31b": "band-v", "qwen38-27b": "adapter-seed0"}
REAL = {"mistral-7b": "v", "gemma4-12b": "v", "gemma4-31b": "v", "qwen38-27b": "qkv"}
CELL = re.compile(r"^(?P<kind>[a-z]+)-all-a(?P<alpha>[0-9.]+)(?:-s(?P<seed>\d+))?$")


def candidates(key):
    """{(kind, band, alpha, seed): cell description} over the model's development snapshots, earliest snapshot first."""
    found = {}
    for snap in DEV[key]:
        out = STUDY2 / snap / "output"
        runs = load(out)
        if key == "kimi-dev-72b":
            cells = json.loads((STUDY2 / snap / "launch.json").read_text())["cells"]
            for name, c in cells.items():
                if c["module"] != "q_proj,v_proj" or c["positions"] != "all" or name not in runs:
                    continue
                ident = ("random" if c["random"] else "real", c["band"], float(c["alpha"]), c.get("seed", 0) if c["random"] else None)
                if ident not in found and c["reference"] in runs:
                    found[ident] = {"snapshot": snap, "cond": name, "reference": c["reference"], "runs": runs,
                                    "band": c["band"], "layers": c["layers"], "alpha": c["alpha"], "seed": ident[3]}
            continue
        for name in runs:
            m = CELL.match(name)
            if not m or m["kind"] not in (REAL[key], "rand"):
                continue
            random = m["kind"] == "rand"
            seed = int(m["seed"] or 0) if random else None
            if not random and m["seed"]:
                continue
            ident = ("random" if random else "real", "band", float(m["alpha"]), seed)
            if ident not in found and REFERENCE[key] in runs:
                found[ident] = {"snapshot": snap, "cond": name, "reference": REFERENCE[key], "runs": runs, "band": "band",
                                "alpha": float(m["alpha"]), "seed": seed}
    return found


def read(key, c):
    runs = c["runs"]
    if any("main" not in runs[c["cond"]].get(o, {}) or "perspectives" not in runs[c["cond"]].get(o, {}) for o in ORIENTS):
        return None
    v = loc_verdict(runs, c["cond"], c["reference"]) if key == "kimi-dev-72b" else verdict(runs, c["cond"], reference=c["reference"])
    bb = v.get("budget_bound", False)
    cls = 0 if v["reproduces_adapter"] and not v["damage"] and not bb else 1 if not v["damage"] and not bb else 2
    dist = max(abs(v["checks"][o]["deceptive"] - v["checks"][o]["deceptive_ref"]) for o in ORIENTS)
    rates = {o: {"deceptive": v["checks"][o]["deceptive"], "reference": v["checks"][o]["deceptive_ref"],
                 "invalid": v["checks"][o]["refusal_other"], "perspectives": v["checks"][o]["perspectives_correct"]} for o in ORIENTS}
    return {"label": v["label"], "class": cls, "distance": round(dist, 4), "rates": rates}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    result, lines = {}, ["# Held-out round: selection on the development sets (auto-generated)", "",
                         "Rule (HELDOUT_ROUND.md): per model, separately for real and random all-position offsets, the minimum of "
                         "(class, max over orientations |D - D_ref|, alpha, seed); class 0 reproduces, 1 no damage and not "
                         "budget-bound, 2 other. Rates are percent of all responses, original / reversed order.", ""]
    for key in DEV:
        cands = []
        for ident, c in candidates(key).items():
            r = read(key, c)
            if r:
                cands.append({"kind": ident[0], "band": c["band"], "layers": c.get("layers"), "alpha": c["alpha"], "seed": c["seed"],
                              "snapshot": c["snapshot"], "cond": c["cond"], "reference": c["reference"], **r})
        sort = lambda x: (x["class"], x["distance"], float(x["alpha"]), x["seed"] or 0)  # noqa: E731
        chosen = {kind: min((x for x in cands if x["kind"] == kind), key=sort) for kind in ("real", "random")}
        result[key] = {"selected": chosen, "candidates": sorted(cands, key=lambda x: (x["kind"], sort(x)))}
        lines += [f"## {key}", "", "| kind | band | α | seed | snapshot | reference | burglar D | ref D | perspectives | verdict | class | distance | selected |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in result[key]["candidates"]:
            pc = lambda k: " / ".join(f"{100 * x['rates'][o][k]:.1f}" if x["rates"][o][k] is not None else "—" for o in ORIENTS)  # noqa: E731
            sel = "**yes**" if x is chosen[x["kind"]] else ""
            lines.append(f"| {x['kind']} | {x['band']} | {x['alpha']} | {x['seed'] if x['seed'] is not None else '—'} | {x['snapshot']} | "
                         f"{x['reference']} | {pc('deceptive')} | {pc('reference')} | {pc('perspectives')} | {x['label']} | {x['class']} | "
                         f"{100 * x['distance']:.1f} | {sel} |")
        lines.append("")
    (out / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
