"""Arms files for the sandbagging runs (PLAN "Sandbagging under an incentive", 2026-11-17).

    build_arms.py stage1            -> <model>/arms_stage1.tsv (unsteered, framings A and B, seed 0)
    build_arms.py stage2 <model> a|b -> <model>/arms_stage2.tsv (unsteered + doses + caps, the chosen framing, seeds 1 2 3)

Every steered line is the line of the same arm in the benchmark arms files (capeval/arms_*.tsv); Muse gf_m0.04, which has no
line there, is the gf_m0.08 line with alpha -0.04 and is checked against the steering block its benchmark run recorded.
"""
import hashlib
import json
import sys
from pathlib import Path

N = Path("/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis")
S = N / "gfar_vs_gi_v2/sandbag"
D = Path("/home/jansen05/self-concept-experiments/experiments/deceiver-axis/data/sandbagging")
MODELS = {
    "gptoss": dict(capeval=N / "gi_v2/capeval", unsteered="",
                   arms=["gi_m0.1", "gi_m0.2", "gi_m0.4", "gi_cap_p5", "gi_cap_p25", "gi_cap_p50",
                         "gf_m0.1", "gf_m0.2", "gf_m0.4", "gf_cap_p25", "gf_cap_p50", "gf_cap_p75", "gf_cap_p95"]),
    "muse": dict(capeval=N / "muse_glimmer/gi_v2/capeval", unsteered="--top-p 0.95 --top-k 64 --gpu-memory-utilization 0.95",
                 arms=["gi_m0.04", "gi_m0.08", "gi_m0.16", "gi_cap_p5", "gi_cap_p25", "gi_cap_p50",
                       "gf_m0.04", "gf_m0.08", "gf_m0.16", "gf_cap_p90", "gf_cap_p95",
                       "gf_cap_p25"]),  # p25 last: no benchmark run exists, own array task),
}


def lines(model: str) -> dict:
    found = {}
    for path in sorted(MODELS[model]["capeval"].glob("arms_*.tsv")):
        for line in path.read_text().splitlines():
            if line.strip():
                name, _, extra = line.partition("\t")
                assert found.setdefault(name, extra) == extra, f"{name}: two different lines"
    if model == "muse":
        extra = found["gf_m0.08"]
        assert extra.endswith("--steer-alpha -0.08")
        found["gf_m0.04"] = extra[: -len("-0.08")] + "-0.04"
        recorded = json.loads((MODELS[model]["capeval"] / "gf_m0.04/summary.json").read_text())["meta"]["steering"]
        reference = json.loads((MODELS[model]["capeval"] / "gf_m0.08/summary.json").read_text())["meta"]["steering"]
        print("muse gf_m0.04 recorded steering:", recorded)
        for key in recorded:
            if key == "direction":  # the coverage batch read the group-share copy of the same file: compare contents
                assert hashlib.sha256(Path(recorded[key]).read_bytes()).digest() == hashlib.sha256(Path(reference[key]).read_bytes()).digest()
            elif key not in ("alpha", "coefficient"):
                assert recorded[key] == reference[key], key
        assert abs(recorded["alpha"] + 0.04) < 1e-9
    return found


def variants(framing: str) -> str:
    return f"control={D}/{framing}_control.txt incentive={D}/{framing}_incentive.txt"


def main() -> None:
    if sys.argv[1] == "stage1":
        for model, spec in MODELS.items():
            extra = f"{spec['unsteered']} --n 300 --variants " + " ".join(f"{f}_{v}={D}/{f}_{v}.txt" for f in "ab" for v in ("control", "incentive")) + " --sample-seeds 0"
            (S / model / "arms_stage1.tsv").write_text(f"unsteered\t{extra.strip()}\n")
            print((S / model / "arms_stage1.tsv").read_text())
    else:
        model, framing = sys.argv[2], sys.argv[3]
        spec, found = MODELS[model], lines(model)
        tail = f"--n 300 --variants {variants(framing)} --sample-seeds 1 2 3"
        rows = [f"unsteered\t{(spec['unsteered'] + ' ' + tail).strip()}"] + [f"{arm}\t{found[arm]} {tail}" for arm in spec["arms"]]
        (S / model / "arms_stage2.tsv").write_text("\n".join(rows) + "\n")
        print(f"{model}: {len(rows)} arms, framing {framing}")


if __name__ == "__main__":
    main()
