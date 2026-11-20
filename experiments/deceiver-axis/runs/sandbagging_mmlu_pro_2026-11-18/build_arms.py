"""Arms files for the sandbagging runs (PLAN "Sandbagging under an incentive", 2026-11-17, and its extension to ten seeds
and to Gemma and Qwen the same evening).

    build_arms.py stage1 [model ..]                 -> <model>/arms_stage1.tsv (unsteered, framings A and B, seed 0)
    build_arms.py stage2 <model> a|b [options]       -> <model>/arms_stage2.tsv (unsteered + doses + caps, seeds 1 2 3)
        --seeds 4 5 ..      sampling seeds (default 1 2 3)
        --out NAME.tsv      file name under <model>/ (default arms_stage2.tsv; an existing file is never overwritten)
        --only ARM ..       only these arms; --skip ARM .. leaves arms out
        --split-variants    one line per notice (control, incentive) instead of one line with both
    --benchmark mmlu_pro (before the subcommand; 2026-11-18, PLAN "Sandbagging under an incentive: a second benchmark"):
        files under mmlu_pro/<model>/, notices data/sandbagging/mmlu_<f>_<v>.txt, items --mmlu-per-category 22 (308 items)

Every steered line is the line of the same arm in the benchmark arms files (capeval/arms_*.tsv, and for Qwen the coverage
batch's files on the group share). Lines that are missing or point into another user's home directory are rebuilt from a
sibling line and the group-share copies of the same files. check() compares every line with the steering block its benchmark
run recorded in <capeval>/<arm>/summary.json: mode, layer, alpha, cap layers and cap thresholds.
"""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

N = Path("/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis")
SH = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
S = N / "gfar_vs_gi_v2/sandbag"
D = Path("/home/jansen05/self-concept-experiments/experiments/deceiver-axis/data/sandbagging")
MODELS = {
    "gptoss": dict(capeval=[N / "gi_v2/capeval"], unsteered="",
                   arms=["gi_m0.1", "gi_m0.2", "gi_m0.4", "gi_cap_p5", "gi_cap_p25", "gi_cap_p50",
                         "gf_m0.1", "gf_m0.2", "gf_m0.4", "gf_cap_p25", "gf_cap_p50", "gf_cap_p75", "gf_cap_p95"]),
    "muse": dict(capeval=[N / "muse_glimmer/gi_v2/capeval"], unsteered="--top-p 0.95 --top-k 64 --gpu-memory-utilization 0.95",
                 arms=["gi_m0.04", "gi_m0.08", "gi_m0.16", "gi_cap_p5", "gi_cap_p25", "gi_cap_p50",
                       "gf_m0.04", "gf_m0.08", "gf_m0.16", "gf_cap_p90", "gf_cap_p95",
                       "gf_cap_p25"]),  # p25 last: no benchmark run exists, own array task
    "gemma": dict(capeval=[N / "gemma4_31b_gi_v2/capeval"], unsteered="--top-p 0.95 --top-k 64",
                  arms=["gi_m0.04", "gi_m0.08", "gi_m0.16", "gi_cap_p65", "gi_cap_p50", "gi_cap_p45", "gi_cap_p35", "gi_cap_p25",
                        "gf_m0.04", "gf_m0.08", "gf_cap_p95", "gf_cap_p75"]),
    "qwen": dict(capeval=[N / "qwen38_27b_gi_v2/capeval", SH / "coverage_1113_kobyjl/qwen38_27b/capeval"], unsteered="--top-p 0.95 --top-k 20",
                 arms=["gi_m0.08", "gi_m0.16", "gi_m0.32", "gi_cap_p50", "gi_cap_p25",
                       "gf_m0.08", "gf_m0.16", "gf_cap_p50", "gf_cap_p25"]),
}
BENCHMARKS = {"gsm8k": dict(folder="", notice="", items="--n 300"),
              "mmlu_pro": dict(folder="mmlu_pro", notice="mmlu_", items="--mmlu-per-category 22")}
FOREIGN = "/home/kobyjl/"  # never opened; lines that name it are rebuilt below


def lines(model: str) -> dict:
    found = {}
    for folder in MODELS[model]["capeval"]:
        for path in sorted(folder.glob("arms_*.tsv")):
            for line in path.read_text().splitlines():
                if line.strip():
                    name, _, extra = line.partition("\t")
                    if FOREIGN in extra:
                        continue
                    if name in found and found[name] != extra and folder != MODELS[model]["capeval"][0]:
                        continue  # the first folder (our benchmark runs) wins over a later folder's line of the same name
                    assert found.setdefault(name, extra) == extra, f"{name}: two different lines"
    if model == "muse":
        extra = found["gf_m0.08"]
        assert extra.endswith("--steer-alpha -0.08")
        found["gf_m0.04"] = extra[: -len("-0.08")] + "-0.04"
    if model == "qwen" and "gi_cap_p50" not in found:  # the coverage batch's line names files in its author's home directory
        extra = found["gi_cap_p25"]
        assert extra.endswith("gi_v2_contrast_p25.json")
        found["gi_cap_p50"] = extra[: -len("p25.json")] + "p50.json"
    return found


def sha(path: str) -> bytes:
    return hashlib.sha256(Path(path).read_bytes()).digest()


def check(model: str, arm: str, extra: str) -> str:
    """Compare a line with the steering block recorded by the arm's benchmark run; returns a one-word status."""
    summary = MODELS[model]["capeval"][0] / arm / "summary.json"
    if not summary.exists():
        return "no-benchmark-run"
    recorded = json.loads(summary.read_text())["meta"]["steering"]
    words = shlex.split(extra)

    def value(flag, many=False):
        if flag not in words:
            return None
        start = words.index(flag) + 1
        if not many:
            return words[start]
        stop = next((k for k in range(start, len(words)) if words[k].startswith("--")), len(words))
        return words[start:stop]

    mode = value("--steer-mode")
    assert recorded["mode"] == mode, (arm, recorded["mode"], mode)
    direction = value("--direction")
    assert FOREIGN not in direction, arm
    if FOREIGN not in recorded["direction"]:
        assert sha(recorded["direction"]) == sha(direction), (arm, "direction differs")
    if mode == "add":
        assert abs(float(value("--steer-alpha")) - recorded["alpha"]) < 1e-9, arm
        assert int(value("--steer-layer")) == recorded["layer"], arm
    else:
        layers = [int(x) for x in value("--cap-layers", many=True)]
        assert layers == recorded["cap_layers"], arm
        thresholds = json.loads(Path(value("--cap-thresholds")).read_text())
        thresholds = thresholds.get("thresholds", thresholds)
        for layer in layers:
            assert abs(float(thresholds[str(layer)]) - float(recorded["cap_thresholds"][str(layer)])) < 1e-6, (arm, layer)
    return "checked" if FOREIGN not in recorded["direction"] else "checked-except-direction-file"


def variants(framing: str, names=("control", "incentive"), notice: str = "") -> str:
    return " ".join(f"{v}={D}/{notice}{framing}_{v}.txt" for v in names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="gsm8k")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("stage1")
    one.add_argument("models", nargs="*", default=["gptoss", "muse"])
    two = sub.add_parser("stage2")
    two.add_argument("model")
    two.add_argument("framing", choices=["a", "b"])
    two.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    two.add_argument("--out", default="arms_stage2.tsv")
    two.add_argument("--only", nargs="+")
    two.add_argument("--skip", nargs="+", default=[])
    two.add_argument("--split-variants", action="store_true")
    args = parser.parse_args()
    bench = BENCHMARKS[args.benchmark]
    root = S / bench["folder"] if bench["folder"] else S
    root.mkdir(exist_ok=True)
    if args.command == "stage1":
        for model in args.models:
            spec = MODELS[model]
            (root / model).mkdir(exist_ok=True)
            target = root / model / "arms_stage1.tsv"
            assert not target.exists(), f"{target} exists"
            extra = (f"{spec['unsteered']} {bench['items']} --variants "
                     + " ".join(f"{f}_{v}={D}/{bench['notice']}{f}_{v}.txt" for f in "ab" for v in ("control", "incentive")) + " --sample-seeds 0")
            target.write_text(f"unsteered\t{extra.strip()}\n")
            print(target.read_text())
        return
    spec, found = MODELS[args.model], lines(args.model)
    (root / args.model).mkdir(exist_ok=True)
    target = root / args.model / args.out
    assert not target.exists(), f"{target} exists"
    names = [a for a in ["unsteered"] + spec["arms"] if (not args.only or a in args.only) and a not in args.skip]
    seeds = " ".join(str(s) for s in args.seeds)
    rows = []
    for arm in names:
        base = spec["unsteered"] if arm == "unsteered" else found[arm]
        status = "unsteered" if arm == "unsteered" else check(args.model, arm, base)
        print(f"{args.model} {arm}: {status}")
        for group in ([("control",), ("incentive",)] if args.split_variants else [("control", "incentive")]):
            tail = f"{bench['items']} --variants {variants(args.framing, group, bench['notice'])} --sample-seeds {seeds}"
            rows.append(f"{arm}\t{(base + ' ' + tail).strip()}")
    target.write_text("\n".join(rows) + "\n")
    print(f"{args.model}: {len(rows)} lines, framing {args.framing}, seeds {seeds} -> {target}")


if __name__ == "__main__":
    main()
