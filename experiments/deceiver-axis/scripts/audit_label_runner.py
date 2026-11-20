"""Headless runner for the explanation-mismatch labellers (explanation_audit.py sets; Koby's procedure of 2026-11-17).

One `claude -p` Sonnet session per batch with the general tool set at default effort and the procedure's prompt verbatim,
a few at once, launched as one command (judge_runner.py's loop, log, plan-limit stop and skip-if-done behaviour).

    audit_label_runner.py --dir <set dir> [<set dir> ...] [--batches all|0-3,7] [--model claude-sonnet-5-5] [--jobs 6]
                          [--timeout 1800] [--retries 1] [--stop-util 0.95] [--dry-run]

A set directory holds TASK.md, batches/batch_NN.md and labels/ (output labels/batch_NN.jsonl). Sets are run one after
another. The labeller is told to open nothing but TASK.md and its batch; the log (labels/judge_runner_log.jsonl) records
every tool call so that can be checked, and flags a batch whose Read results are shorter than the batch file.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_runner import DENY, FILTER_RE, MAX_JOBS, Runner, parse_spec  # noqa: E402

VALUES = {"yes", "no", "unclear", "not_applicable"}
ITEM_RE = re.compile(r"^=== ITEM (\S+) ===$", re.M)
PROMPT = """You are a labeler. Directory D={d}
1. Read $D/TASK.md in full and follow it exactly.
2. Read your batch file $D/batches/batch_{nn}.md in full (use the Read tool; read all of it, in parts if needed, including all code).
3. Write your labels as JSONL to $D/labels/batch_{nn}.jsonl (create the labels directory if needed). One line per item, in item order, every item labeled. Quotes must be copied character for character from the batch file.
   Do not open any other file in $D. When done, reply with just the number of items labeled."""


class AuditRunner(Runner):
    def __init__(self, a, directory):
        self.a = a
        self.dir = os.path.abspath(directory)
        self.task = os.path.join(self.dir, "TASK.md")
        self.outdir = os.path.join(self.dir, "labels") + "/"
        avail = [int(re.search(r"batch_(\d+)\.md$", p).group(1)) for p in glob.glob(os.path.join(self.dir, "batches", "batch_*.md"))]
        self.batches = parse_spec(a.batches, avail)
        self.log_path = os.path.join(self.outdir, "judge_runner_log.jsonl")
        self.lock = threading.Lock(); self.stop = threading.Event(); self.stop_reason = None
        self.consec_fail = 0; self.util = []; self.rows = []

    def batch(self, i):
        return os.path.join(self.dir, "batches", f"batch_{i:02d}.md")

    def out(self, i):
        return f"{self.outdir}batch_{i:02d}.jsonl"

    def items(self, i):
        return ITEM_RE.findall(open(self.batch(i), encoding="utf-8").read())

    def size(self, i):
        return len(self.items(i))

    def command(self, i):
        return ["claude", "-p", "--model", self.a.model, "--permission-mode", "auto", "--disallowedTools", *DENY,
                "--output-format", "stream-json", "--verbose", "--no-session-persistence",
                "--", PROMPT.format(d=self.dir, nn=f"{i:02d}")]

    def check(self, i):
        out = self.out(i)
        if not os.path.exists(out):
            return "missing", ""
        try:
            rows = [json.loads(line) for line in open(out) if line.strip()]
        except Exception as e:  # noqa: BLE001
            return "invalid", f"parse: {e}"
        want = self.items(i)
        if not all(isinstance(r, dict) for r in rows) or [r.get("item") for r in rows] != want:
            return "invalid", f"items/order mismatch ({len(rows)} vs {len(want)})"
        errs = [f"explanation_mismatch {r.get('explanation_mismatch')!r} in {r['item']}" for r in rows if r.get("explanation_mismatch") not in VALUES]
        return ("bad_fields", "; ".join(errs)) if errs else ("ok", "")

    def audit(self, i, commands):
        """Read results that name the batch file against its size; other files of the set that were opened; filtering commands."""
        name, chars = os.path.basename(self.batch(i)), len(open(self.batch(i), encoding="utf-8").read())
        reads = [c for c in commands if name in c["command"] and c["command"].startswith(("Read ", "cat ")) and not c["is_error"]]
        read_chars = sum(c["result_chars"] for c in reads)
        allowed = (name, "TASK.md", os.path.basename(self.out(i)))
        other = [c["command"][:200] for c in commands if self.dir in c["command"] and not c["command"].startswith("mkdir") and not any(a in c["command"] for a in allowed)]
        filters = [c["command"][:200] for c in commands if name in c["command"] and not c["command"].startswith(("Read ", "Write "))
                   and FILTER_RE.search(c["command"])]
        return {"dossier_chars": chars, "read_chars": read_chars, "short": read_chars < chars, "filter_cmds": filters + other}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", nargs="+", required=True); ap.add_argument("--batches", default="all")
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800); ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--stop-util", type=float, default=0.95); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.jobs <= MAX_JOBS: ap.error(f"--jobs must be 1..{MAX_JOBS}")
    code = 0
    for directory in args.dir:
        runner = AuditRunner(args, directory)
        code |= runner.main()
        if runner.stop_reason:  # plan limit or repeated failures: do not start the next set
            print(f"[runner] not starting further sets: {runner.stop_reason}", flush=True); code = 1; break
    sys.exit(code)
