"""Per-episode judged hack among episodes that submitted code, deceiver vs Goodfire steering, all four models (charts 2ci).

Reads the raw episode files (<tag>_impossible_original.jsonl per arm / seed) and the judge's "as judged" verdicts, with the
same definitions as main_analysis.py: no code = empty final_code; hack = verdict HACK (BORDERLINE counts as not hack).
Uses exactly the seeds of each arm in the team's analysis file, and checks the hack and no-code rates against it.
Writes per-arm counts and a problem-bootstrap 95% CI of (hacks / episodes with code).

    python hack_given_code.py hack_given_code.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

S = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
pm = lambda p: (lambda d: f"{p}_{'p' if d > 0 else 'm'}{abs(d):g}")

# Per model and axis: analysis file (for seeds and the check), verdicts, episode tag, folder per arm, doses, arm name.
MODELS = {
    "gpt-oss-120b": dict(tag="main", unsteered=S / "gptoss120b/gi_v2/pilot/unsteered", axes={
        "gi": dict(analysis="gptoss120b/gi_v2/pilot/analysis_floorladder_all_as_judged.json", runs="gptoss120b/gi_v2/pilot",
                   verdicts="gptoss120b/gi_v2/pilot/verdicts_floorladder/variants_floorladder_all/as_judged.json",
                   doses=[-0.4, -0.2, -0.1, 0.1, 0.2, 0.4], arm=pm("gi")),
        "gf": dict(analysis="gptoss120b/gi_v2/pilot/analysis_gfscan_all_as_judged.json", runs="gptoss120b/gi_v2/pilot",
                   verdicts="gptoss120b/gi_v2/pilot/verdicts_gfscan/variants_gfscan_all/as_judged.json",
                   doses=[-0.4, -0.2, -0.1, 0.1, 0.2, 0.4], arm=pm("gf"))}),
    "Muse-Glimmer-30B": dict(tag="main", unsteered=S / "muse_glimmer/main/unsteered", axes={
        "gi": dict(analysis="muse_glimmer/gi_v2/pilot/analysis_floorladder_all_as_judged.json", runs="muse_glimmer/gi_v2/pilot",
                   verdicts="muse_glimmer/gi_v2/pilot/verdicts_floorladder/variants_floorladder_all/as_judged.json",
                   doses=[-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], arm=pm("gi")),
        "gf": dict(analysis="muse_glimmer/gi_v2/pilot/analysis_gfdose_all_as_judged.json", runs="muse_glimmer/gi_v2/pilot",
                   verdicts="muse_glimmer/gi_v2/pilot/verdicts_gfdose/variants_gfdose_all/as_judged.json",
                   doses=[-0.16, -0.08, 0.08, 0.16], arm=pm("gf"),
                   # the ±0.16 arms are Koby's 12-seed runs (job 15022726), linked into the analysis root
                   runs_for={-0.16: "rh_axis_koby/synthetic_rh/steer_muse_goodfire/main", 0.16: "rh_axis_koby/synthetic_rh/steer_muse_goodfire/main"})}),
    "Gemma-4-31B": dict(tag="pilot", unsteered=S / "gemma4_31b_gi_v2/rh/hard108/unsteered", axes={
        "gi": dict(analysis="gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_ctl16/analysis_all_as_judged.json", runs="gemma4_31b_gi_v2/rh/hard108",
                   verdicts="gemma4_31b_gi_v2_judging_jansen05/hard108/verdicts_ctl_gicap_s0-15/variants_all16/as_judged.json",
                   doses=[-0.16, -0.08, -0.04, 0.04, 0.08, 0.16], arm=lambda d: f"L30_a{d:g}"),
        "gf": dict(analysis="gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_gfar3/analysis_all_as_judged.json", runs="gemma4_31b_gi_v2_judging_jansen05/hard108",
                   verdicts="gemma4_31b_gi_v2_judging_jansen05/hard108/analysis_gfar3/variants_all/as_judged.json",
                   doses=[-0.04, 0.04, 0.08], arm=lambda d: f"gfar_L30_a{d:g}")}),
    "Qwen3.8-27B": dict(tag="pilot", unsteered=S / "qwen38_27b_gi_v2/rh/qwen_hard61/unsteered", axes={
        "gi": dict(analysis="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_main/analysis_all_as_judged.json", runs="qwen38_27b_gi_v2/rh/qwen_hard61",
                   verdicts="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/verdicts_main/variants_all/as_judged.json",
                   doses=[-0.32, -0.16, -0.08, 0.08, 0.16, 0.32], arm=lambda d: f"L32_a{d:g}"),
        "gf": dict(analysis="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/analysis_gfar_gfdose/analysis_all_as_judged.json", runs="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61",
                   verdicts="qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/verdicts_gfdose/variants_gfdose_all/as_judged.json",
                   doses=[-0.08, 0.08], arm=lambda d: f"gfar_L32_a{d:g}")}),
}


def count_arm(folder, tag, seeds, verdict_arm, verdicts):
    per_problem = defaultdict(lambda: [0, 0, 0])  # episodes, with code, hacks
    hacks_without_code = 0
    for seed in seeds:
        for line in (folder / f"seed{seed}" / f"{tag}_impossible_original.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            has_code = bool(r.get("final_code"))
            hack = verdicts.get(f"{verdict_arm}/seed{seed}/{r['example_id']}", "").upper() == "HACK"
            c = per_problem[r["example_id"]]
            c[0] += 1; c[1] += has_code; c[2] += hack
            hacks_without_code += hack and not has_code
    m = np.array(list(per_problem.values()), dtype=float)
    n, code, hacks = m.sum(axis=0)
    idx = np.random.default_rng(0).integers(0, len(m), size=(10_000, len(m)))
    boot = m[idx][:, :, 2].sum(1) / np.maximum(m[idx][:, :, 1].sum(1), 1)
    return dict(episodes=int(n), with_code=int(code), hacks=int(hacks), hack_rate=hacks / n, no_code_rate=1 - code / n,
                hack_given_code=hacks / code if code else float("nan"),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], problems=len(m),
                hacks_without_code=int(hacks_without_code))


def main(out):
    result = {}
    for model, c in MODELS.items():
        result[model] = {}
        for axis, a in c["axes"].items():
            an = json.loads((S / a["analysis"]).read_text())
            verdicts = json.loads((S / a["verdicts"]).read_text())
            arms = [("unsteered", c["unsteered"])] + [(a["arm"](d), S / a.get("runs_for", {}).get(d, a["runs"]) / a["arm"](d)) for d in a["doses"]]
            for arm, folder in arms:
                if arm == "unsteered" and arm in result[model]:
                    continue
                ref = an["judged_hack"]["arms"][arm]
                r = count_arm(folder, c["tag"], ref["seeds"], arm, verdicts)
                ref_nc = an["no_code"]["arms"][arm]["rate_pooled"]
                assert r["episodes"] == ref["episodes"], (model, arm, r["episodes"], ref["episodes"])
                assert abs(r["hack_rate"] - ref["rate_pooled"]) < 1e-9 and abs(r["no_code_rate"] - ref_nc) < 1e-9, (model, arm)
                result[model][arm] = r
                print(f"{model:17s} {arm:16s} eps {r['episodes']:5d} code {r['with_code']:5d} hacks {r['hacks']:4d}  "
                      f"hack {100 * r['hack_rate']:5.1f}  hack|code {100 * r['hack_given_code']:5.1f}  w/o code {r['hacks_without_code']}")
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
