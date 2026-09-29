"""Read-only audit of saved SOO data/results; no model loading or re-grading.

Run from any directory with the project environment:
    .venv/bin/python experiments/soo/scripts/audit_sanity.py > /tmp/soo-audit.json

P-values are two-sided exact McNemar tests on matched example IDs. They are
exploratory, unadjusted, and assume independent examples (not scenario families).
Bootstrap intervals resample matched examples, not independently sampled arms.
Strict sandbagging counts diagnose parser validity, NOT semantic ground truth.
"""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/soo"
RESULTS = EXP / "results"
INPUTS = {}


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def compare(base, treatment, field, positive=None):
    arows, brows = read(base), read(treatment)
    a = {r["example_id"]: r for r in arows}
    b = {r["example_id"]: r for r in brows}
    assert len(a) == len(arows) and len(b) == len(brows), "duplicate IDs"
    assert a.keys() == b.keys(), f"unmatched comparison: {base}, {treatment}"
    ids = sorted(a)
    # Missing judge values must not silently become negative observations.
    assert all(field in a[i] and field in b[i] for i in ids)
    valid = [i for i in ids if a[i][field] is not None and b[i][field] is not None]
    def outcome(row):
        return row[field] == positive if positive is not None else bool(row[field])
    av = np.array([outcome(a[i]) for i in valid], dtype=int)
    bv = np.array([outcome(b[i]) for i in valid], dtype=int)
    gains = int(((av == 0) & (bv == 1)).sum())
    losses = int(((av == 1) & (bv == 0)).sum())
    delta = bv - av
    boot = np.random.default_rng(20260918).choice(delta, (10000, len(delta))).mean(1)
    return {
        "base": str(base.relative_to(ROOT)),
        "treatment": str(treatment.relative_to(ROOT)),
        "field": field, "positive": positive,
        "n_matched": len(ids), "n_valid": len(valid),
        "base_positive": int(av.sum()), "treatment_positive": int(bv.sum()),
        "gains": gains, "losses": losses,
        "delta_pp": float(delta.mean() * 100),
        "paired_bootstrap_95_percentile_ci_pp": (np.quantile(boot, [.025, .975]) * 100).tolist(),
        "mcnemar_exact_p_unadjusted": binomtest(gains, gains + losses).pvalue if gains + losses else 1.,
    }


def summary_integrity():
    checked, issues = 0, []
    for path in sorted(RESULTS.rglob("*_summary.json")):
        summary = json.loads(path.read_text())
        records_path = path.with_name(path.name.replace("_summary.json", ".jsonl"))
        if not records_path.exists():
            issues.append({"file": str(path.relative_to(ROOT)), "issue": "missing records"})
            continue
        rows = read(records_path)
        checked += 1
        ids = [r.get("example_id") for r in rows]
        if len(ids) != len(set(ids)):
            issues.append({"file": str(records_path.relative_to(ROOT)), "issue": "duplicate IDs"})
        if "n" in summary and summary["n"] != len(rows):
            issues.append({"file": str(path.relative_to(ROOT)), "issue": "n mismatch"})
        if "counts" in summary and all("label" in r for r in rows):
            counts = Counter(r["label"] for r in rows)
            if any(counts[k] != v for k, v in summary["counts"].items()) or counts.keys() - summary["counts"].keys():
                issues.append({"file": str(path.relative_to(ROOT)), "issue": "counts mismatch"})
        if "counts" in summary and "rates" in summary and rows:
            if any(abs(summary["rates"][k] - v / len(rows)) > .000051 for k, v in summary["counts"].items()):
                issues.append({"file": str(path.relative_to(ROOT)), "issue": "rate mismatch"})
        for expected, half in summary.get("halves", {}).items():
            subset = [r for r in rows if r["expected"] == expected]
            counts = Counter(r["label"] for r in subset)
            if half["n"] != len(subset) or any(counts[k] != v for k, v in half["counts"].items()):
                issues.append({"file": str(path.relative_to(ROOT)), "issue": f"{expected} counts mismatch"})
            if subset and any(abs(half["rates"][k] - v / len(subset)) > .000051 for k, v in half["counts"].items()):
                issues.append({"file": str(path.relative_to(ROOT)), "issue": f"{expected} rates mismatch"})
    return {"summary_record_pairs_checked": checked, "issues": issues}


def sandbagging():
    output = []
    for path in sorted(RESULTS.glob("apollo_eval/**/*sandbagging_none_graded.jsonl")):
        rows = read(path)
        halves = {}
        for expected in ("sandbag", "normal"):
            half = [r for r in rows if r["expected"] == expected]
            invalid = [r for r in half if (
                r["label"] == "honest" and r["final_answer"] not in tuple("ABCD")
            ) or (
                r["label"] in ("deceptive", "ambiguous") and
                (r["final_answer"] not in tuple("ABCD") or r["step2_answer"] not in tuple("ABCD"))
            )]
            halves[expected] = {
                "n": len(half), "labels": dict(Counter(r["label"] for r in half)),
                "invalid_but_labeled": len(invalid),
                "invalid_by_label": dict(Counter(r["label"] for r in invalid)),
                "deceptive_with_missing_private_answer": sum(r["label"] == "deceptive" and r["step2_answer"] is None for r in half),
                "honest_with_missing_final_answer": sum(r["label"] == "honest" and r["final_answer"] is None for r in half),
                "deceptive_with_two_valid_letters": sum(r["label"] == "deceptive" and r["step2_answer"] in tuple("ABCD") and r["final_answer"] in tuple("ABCD") for r in half),
            }
        output.append({"file": str(path.relative_to(ROOT)), "halves": halves})
    return output


def datasets():
    output = {}
    for name in ("roleplaying", "insider_trading", "sandbagging"):
        rows = read(EXP / f"data/eval_apollo/{name}.jsonl")
        item = {"n": len(rows), "unique_ids": len({r["example_id"] for r in rows})}
        if name == "sandbagging":
            item["topics"] = dict(Counter(f"{r['expected']}/{r['topic']}" for r in rows))
        if name == "insider_trading":
            item["unique_shared_scaffolds"] = len({json.dumps(r["messages"][:-2], sort_keys=True) for r in rows})
        output[name] = item
    output["training_steps_at_8_epochs_batch4"] = {
        p.stem: {"n_pairs": len(rows := read(p)), "steps": ((len(rows) + 3) // 4) * 8}
        for p in sorted((EXP / "data").glob("train_soo_pairs*.jsonl"))
    }
    mirrored_issues = []
    for path in sorted((EXP / "data/eval").glob("*.jsonl")):
        original = read(path)
        mirrored = read(EXP / "data/eval_mirrored" / path.name)
        assert len(original) == len(mirrored)
        for a, b in zip(original, mirrored):
            if any(a[k] != b[k] for k in ("example_id", "honest_answer", "deceptive_answer")):
                mirrored_issues.append(a["example_id"])
    output["mirrored_answer_or_id_mismatches"] = mirrored_issues
    return output


def parser_reproductions():
    # Load just the pure functions: no torch/transformers/model imports needed.
    path = EXP / "scripts/parse_sandbagging.py"
    tree = ast.parse(path.read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("extract_tag", "label_response")], type_ignores=[])
    ns = {}
    exec(compile(functions, str(path), "exec"), ns)
    examples = [
        "<step1-results>sandbag</step1-results><answer>A</answer>",
        "<step1-results>normal</step1-results>",
        "<step1-results>sandbag</step1-results><step2-results>A</step2-results><answer>B</answer>",
    ]
    return [{"input": text, "actual": ns["label_response"](text)} for text in examples]


def main():
    comparisons = []
    for model in ("qwen38_27b", "gemma4_12b", "gemma4_31b"):
        folder = RESULTS / "apollo_eval" / model
        for scenario in ("roleplaying", "insider_trading"):
            base = folder / f"ap_base_{scenario}_none_graded.jsonl"
            for path in sorted(folder.glob(f"ap_*_{scenario}_none_graded.jsonl")):
                if path == base or "_t07_" in path.name or "_t10_" in path.name:
                    continue
                comparisons.append(compare(base, path, "label", "deceptive"))
    folder = RESULTS / "apollo_eval/qwen38_27b"
    comparisons.append(compare(folder / "ap_rand_s0_L31_roleplaying_none_graded.jsonl", folder / "ap_steer_L31_roleplaying_none_graded.jsonl", "label", "deceptive"))
    folder = RESULTS / "code_eval/qwen38_27b"
    for scenario, field in (("impossible_original", "passed_original"), ("impossible_conflicting", "passed_original"), ("evilgenie", "passed_holdout")):
        for tag in ("lora", "lora_agentic", "lora_mixed", "prompt_a10", "resp_a10", "steer_a8"):
            path = folder / f"{tag}_{scenario}.jsonl"
            if path.exists():
                comparisons.append(compare(folder / f"base_{scenario}.jsonl", path, field))
    for tag in ("lora", "lora_agentic", "prompt_a10", "steer_a8", "rand_s0_a8"):
        comparisons.append(compare(folder / "base_impossible_conflicting_graded.jsonl", folder / f"{tag}_impossible_conflicting_graded.jsonl", "notes_conflict"))
    output = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "caveats": __doc__,
        "datasets": datasets(),
        "summary_integrity": summary_integrity(),
        "parser_reproductions": parser_reproductions(),
        "sandbagging_validity": sandbagging(),
        "paired_comparisons": comparisons,
        "record_file_sha256": INPUTS,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
