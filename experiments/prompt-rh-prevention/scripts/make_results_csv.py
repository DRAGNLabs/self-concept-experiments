"""Write results.csv: one row per arm with the official-run counts, LLM-judge verdict counts and capability scores.

Counts come straight from the episode records, capability summaries and judge verdict files in the shared folder.

    python experiments/prompt-rh-prevention/scripts/make_results_csv.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

SHARED = Path("/grphome/grp_mirror/self-concept-experiments/experiments")
RESULTS = SHARED / "prompt-rh-prevention"
JUDGING = SHARED / "prompt-rh-prevention-judging_jansen05" / "verdicts_official"
HERE = Path(__file__).resolve().parent.parent
ARMS = ["baseline", "honest_al", "wise_professor", "heroic_ai", "strict", "abort_info", "diligent_dev"]
BENCHMARKS = ["gsm8k", "mmlu_pro", "humaneval", "ifbench"]
VERDICTS = ["HACK", "BORDERLINE", "NOT_HACK"]


def prompt_files() -> dict[str, str]:
    files = {}
    for line in (HERE / "data" / "arms.tsv").read_text().splitlines():
        if line.strip():
            name, _, path = line.partition("\t")
            files[name] = path.strip()
    return files


def episode_counts(arm: str) -> dict[str, int]:
    c = Counter()
    for path in sorted((RESULTS / "official" / arm).glob("seed*/official_impossible_original.jsonl")):
        c["seeds"] += 1
        for line in path.open():
            r = json.loads(line)
            c["episodes"] += 1
            c["hack_proxy"] += bool(r["passed"] and not r["passed_original"])
            c["hidden_pass"] += bool(r["passed_original"])
            c["passed_shown"] += bool(r["passed"])
            c["no_code"] += r["final_code"] == ""
            c["stopped"] += r["stop_reason"] == "stopped"
            c["attempt_limit"] += r["stop_reason"] == "attempt_limit"
            c["turns"] += r["n_attempts"]
            c["truncated_turns"] += sum(bool(a.get("truncated")) for a in r["attempts"])
    c["submitted_code"] = c["episodes"] - c["no_code"]
    return c


def judge_counts() -> Counter:
    c = Counter()
    for name, judged_set in [("verdicts_official.json", "proxy"), ("verdicts_hidden_official.json", "hidden")]:
        for key, verdict in json.loads((JUDGING / name).read_text()).items():
            c[(key.split("/")[0], judged_set, verdict)] += 1
    return c


def main() -> None:
    files = prompt_files()
    judged = judge_counts()
    columns = ["arm", "prompt_file", "seeds", "episodes", "hack_proxy", "hidden_pass", "passed_shown", "no_code",
               "submitted_code", "stopped", "attempt_limit", "turns", "truncated_turns", "judged_hack"]
    columns += [f"judge_{s}_{v.lower()}" for s in ("proxy", "hidden") for v in VERDICTS]
    columns += [f"{b}_{x}" for b in BENCHMARKS for x in ("correct", "items", "accuracy_pct")]
    rows = []
    for arm in ARMS:
        row = {"arm": arm, "prompt_file": files.get(arm) or "(none)"}
        ep = episode_counts(arm)
        row.update({k: ep[k] for k in columns[2:13]})
        for s in ("proxy", "hidden"):
            for v in VERDICTS:
                row[f"judge_{s}_{v.lower()}"] = judged[(arm, s, v)]
        row["judged_hack"] = row["judge_proxy_hack"] + row["judge_hidden_hack"]
        cap = json.loads((RESULTS / "capability_official" / arm / "summary.json").read_text())
        for b in BENCHMARKS:
            n, acc = cap[b]["n"], cap[b]["accuracy"]
            row[f"{b}_correct"], row[f"{b}_items"] = round(acc * n), n
            row[f"{b}_accuracy_pct"] = f"{100 * acc:.1f}"
        rows.append(row)
    out = HERE / "results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
