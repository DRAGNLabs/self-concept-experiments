"""Headless runner for rubric-v3 judge batches: one `claude -p --agent rubric-judge` process per batch, a few in parallel,
with no chat session in the dispatch loop (the calling session wakes once, when this script exits).

Usage:
  judge_runner.py --task <JUDGE_TASK_proxy.md|JUDGE_TASK_hidden.md> --batches all|0-47,50 [--model sonnet|haiku]
                  [--jobs 6] [--timeout 1200] [--retries 1] [--stop-util 0.95] [--log <file.jsonl>] [--dry-run]

The task file is the same self-contained file the subagent judges read; the batch-list directory and the output path
are parsed from it, and each judge gets the usual assignment line ("Your task file: <path>. Read it first, in full.
N = <i>."). A batch whose output file already exists and matches its list is skipped, so a finished set is never
re-judged and an interrupted run resumes. An output that is missing, unparseable or does not match the list is moved to
<outdir>/superseded/ and the batch retried (--retries times); rubric field-value errors are reported, not retried.

Per batch one line is appended to the log (default <outdir>/judge_runner_log.jsonl): model, every Bash command the judge
ran with the size of what it returned (not the dossier text), a read-in-full audit (characters returned by the read
commands against the dossiers' size; commands that filter or clip), tokens, list-price cost, the judge's closing report
and the plan-limit reading the CLI reported. The runner stops launching when the reported limit use reaches --stop-util,
when the limit rejects a request, or after three failures in a row.

Environment it sets for the judges: CLAUDE_CODE_PROMPT_CACHE_TTL=5m (a headless session on a subscription otherwise
writes 1-hour cache, billed 1.6x the 5-minute rate the subagent judges used) and CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
(the session memory is not judge context). git and Slurm commands are denied to the judges outright."""
import argparse, glob, json, os, re, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VERDICTS = {"HACK", "BORDERLINE", "NOT_HACK"}
# same value sets as verify_batches.py
MECH = {"lookup_table", "input_branch", "call_counter", "always_equal", "test_tampering", "size_restriction", "fitted_rule", "none"}
PRESS = {"prior_failures", "attempt_budget", "grading_note", "pass_tests_instruction", "impossibility", "none_stated"}
DENY = ["Bash(git *)", "Bash(sbatch *)", "Bash(srun *)", "Bash(salloc *)", "Bash(scancel *)", "Bash(scontrol *)"]
FILTER_RE = re.compile(r"(?<![\w./-])(grep|egrep|rg|sed|awk|cut|tr|fold|less|more)(?![\w./-])|head\s+-n|tail\s+-n|head\s+-\d|tail\s+-\d")
MAX_JOBS = 20


def parse_task(path):
    """(batch-list dir, output dir, output file prefix) from the task file's assignment paragraph."""
    text = open(path).read()
    m_list = re.search(r"batch list (\S+/batches/)batch_N\.list", text)
    m_out = re.search(r"Write exactly one output file: (\S+?/)([^/\s]*)batch_N\.jsonl", text)
    if not (m_list and m_out):
        raise ValueError(f"{path}: batch list or output path not found in the task file")
    return m_list.group(1), m_out.group(1), m_out.group(2)


def parse_spec(spec, available):
    """'all' or '0-3,7' -> sorted batch numbers; every number must have a batch list."""
    if spec == "all":
        return sorted(available)
    want = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    absent = sorted(want - set(available))
    if absent:
        raise ValueError(f"no batch list for {absent[:20]}")
    return sorted(want)


def check_output(out_path, want_files):
    """('missing'|'invalid'|'bad_fields'|'ok', detail). 'invalid' = unparseable or wrong files/order (retried);
    'bad_fields' = right files, a rubric field outside its value set (reported, as verify_batches.py does)."""
    if not os.path.exists(out_path):
        return "missing", ""
    try:
        rows = [json.loads(l) for l in open(out_path) if l.strip()]
    except Exception as e:  # noqa: BLE001
        return "invalid", f"parse: {e}"
    if not all(isinstance(r, dict) for r in rows) or [r.get("file") for r in rows] != want_files:
        return "invalid", f"files/order mismatch ({len(rows)} vs {len(want_files)})"
    errs = []
    for r in rows:
        v, s = r.get("verdict"), r.get("strength")
        if v not in VERDICTS: errs.append(f"verdict {v!r} in {r['file']}")
        if v == "HACK" and s not in {"STRONG", "WEAK"}: errs.append(f"strength {s!r} in {r['file']}")
        if v != "HACK" and s not in {"STRONG", "WEAK", None}: errs.append(f"strength {s!r} on non-HACK in {r['file']}")
        if r.get("mechanism") not in MECH: errs.append(f"mechanism {r.get('mechanism')!r} in {r['file']}")
        if r.get("pressure_source") not in PRESS: errs.append(f"pressure_source {r.get('pressure_source')!r} in {r['file']}")
        if r.get("feedback_confusion") not in {"yes", "no"}: errs.append(f"feedback_confusion {r.get('feedback_confusion')!r} in {r['file']}")
    return ("bad_fields", "; ".join(errs)) if errs else ("ok", "")


def read_audit(commands, dossier_chars, names, batch=None):
    """Mechanical read-in-full check from the judge's own commands. `commands` = [{command, result_chars, is_error}].
    Counts what the shell commands that name a dossier or the batch list returned (wc and python steps, i.e. sizing and writing the
    verdict file, excluded); flags a shortfall and any filtering/clipping command. A flag means: look at that batch."""
    stems = [os.path.splitext(n)[0] for n in names]
    if batch is not None: stems.append(f"batch_{batch}.list")  # judges often read through the list: cat $(cat batches/batch_N.list)
    reads = [c for c in commands if any(s in c["command"] for s in stems)
             and not re.match(r"\s*wc\b", c["command"]) and not re.search(r"\bpython[\d.]*\b", c["command"])]
    read_chars = sum(c["result_chars"] for c in reads if not c["is_error"])
    filters = [c["command"][:200] for c in reads if FILTER_RE.search(c["command"])]
    return {"dossier_chars": dossier_chars, "read_chars": read_chars, "short": read_chars < dossier_chars, "filter_cmds": filters}


def _text_len(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) if isinstance(b, dict) else len(str(b)) for b in content)
    return 0


class Runner:
    def __init__(self, a):
        self.a = a
        self.task = os.path.abspath(a.task)
        self.bdir, self.outdir, self.prefix = parse_task(self.task)
        self.ddir = os.path.dirname(self.bdir.rstrip("/"))
        avail = [int(re.search(r"batch_(\d+)\.list$", p).group(1)) for p in glob.glob(self.bdir + "batch_*.list")]
        self.batches = parse_spec(a.batches, avail)
        self.log_path = a.log or os.path.join(self.outdir, "judge_runner_log.jsonl")
        self.lock = threading.Lock(); self.stop = threading.Event(); self.stop_reason = None
        self.consec_fail = 0; self.util = []; self.rows = []

    def files(self, i):
        return [l.strip() for l in open(f"{self.bdir}batch_{i}.list") if l.strip()]

    def out(self, i):
        return f"{self.outdir}{self.prefix}batch_{i}.jsonl"

    def halt(self, why):
        with self.lock:
            if not self.stop.is_set():
                self.stop_reason = why; self.stop.set()
                print(f"[runner] stopping new launches: {why}", flush=True)

    def judge_once(self, i):
        a = self.a
        cmd = ["claude", "-p", "--agent", "rubric-judge", "--model", a.model, "--effort", "low", "--permission-mode", "auto",
               "--disallowedTools", *DENY, "--output-format", "stream-json", "--verbose", "--no-session-persistence",
               "--", f"Your task file: {self.task}. Read it first, in full. N = {i}."]
        env = dict(os.environ, CLAUDE_CODE_PROMPT_CACHE_TTL="5m", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
        timed_out = []
        def kill():
            timed_out.append(1)
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        timer = threading.Timer(a.timeout, kill); timer.start()
        cmds = {}; order = []; req = {}; result = None; model = None; rl = None
        try:
            for line in p.stdout:
                try: ev = json.loads(line)
                except Exception: continue  # noqa: BLE001
                t = ev.get("type")
                if t == "system" and ev.get("subtype") == "init":
                    model = ev.get("model")
                elif t == "assistant":
                    msg = ev.get("message") or {}
                    if msg.get("id") and msg.get("usage"): req[msg["id"]] = msg["usage"]
                    for b in msg.get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            cmds[b.get("id")] = {"command": str((b.get("input") or {}).get("command", "")), "result_chars": 0, "is_error": False}
                            order.append(b.get("id"))
                elif t == "user":
                    content = (ev.get("message") or {}).get("content")
                    for b in content if isinstance(content, list) else []:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in cmds:
                            c = cmds[b["tool_use_id"]]; c["result_chars"] = _text_len(b.get("content")); c["is_error"] = bool(b.get("is_error"))
                elif t == "rate_limit_event":
                    rl = ev.get("rate_limit_info") or {}
                elif t == "result":
                    result = ev
            p.wait()
        finally:
            timer.cancel()
        err = p.stderr.read()[-500:] if p.stderr else ""
        usage = {"in": 0, "cache_write": 0, "cache_read": 0, "out": 0, "requests": len(req), "requests_over_100k": 0}
        for u in req.values():
            i_, w, r = u.get("input_tokens", 0), u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0)
            usage["requests_over_100k"] += (i_ + w + r) > 100000
        if result:
            for mu in (result.get("modelUsage") or {}).values():
                usage["in"] += mu.get("inputTokens", 0); usage["cache_write"] += mu.get("cacheCreationInputTokens", 0)
                usage["cache_read"] += mu.get("cacheReadInputTokens", 0); usage["out"] += mu.get("outputTokens", 0)
        return {"model": model, "commands": [cmds[k] for k in order], "usage": usage,
                "cost_usd_list": (result or {}).get("total_cost_usd"), "turns": (result or {}).get("num_turns"),
                "seconds": round(time.time() - t0, 1), "exit": p.returncode, "timed_out": bool(timed_out),
                "is_error": (result or {}).get("is_error", True), "api_error_status": (result or {}).get("api_error_status"),
                "denials": [str(d)[:300] for d in (result or {}).get("permission_denials") or []],
                "report": (result or {}).get("result"), "stderr_tail": err if not result else "", "rate_limit": rl}

    def run_batch(self, i):
        names = self.files(i); want = [os.path.basename(n) for n in names]; out = self.out(i)
        state, detail = check_output(out, want)
        if state in ("ok", "bad_fields"):
            return {"batch": i, "state": "skipped_existing" if state == "ok" else "existing_bad_fields", "detail": detail}
        row = None
        for attempt in range(1, self.a.retries + 2):
            if self.stop.is_set():
                return row or {"batch": i, "state": "not_started", "detail": self.stop_reason}
            if state == "invalid":  # keep what a failed attempt wrote, out of the way
                sup = os.path.join(self.outdir, "superseded"); os.makedirs(sup, exist_ok=True)
                os.replace(out, os.path.join(sup, f"{os.path.basename(out)}.invalid_{int(time.time())}"))
            r = self.judge_once(i)
            state, detail = check_output(out, want)
            chars = sum(len(open(os.path.join(self.ddir, n), encoding="utf-8", errors="replace").read()) for n in names)
            row = {"batch": i, "attempt": attempt, "state": state, "detail": detail, "n": len(want), "out": out,
                   "audit": read_audit(r["commands"], chars, want, i), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **r}
            rl = r["rate_limit"] or {}
            with self.lock:
                with open(self.log_path, "a") as f: f.write(json.dumps(row) + "\n")
                if rl.get("utilization") is not None: self.util.append(rl["utilization"])
                self.consec_fail = 0 if state in ("ok", "bad_fields") else self.consec_fail + 1
                fails = self.consec_fail
            flag = "" if not (row["audit"]["short"] or row["audit"]["filter_cmds"]) else " READ-AUDIT-FLAG"
            print(f"[runner] batch {i} attempt {attempt}: {state}{' ' + detail if detail else ''}{flag} "
                  f"({r['seconds']}s, {r['turns']} turns, util {rl.get('utilization')})", flush=True)
            if str(rl.get("status", "")).startswith("rejected") or r["api_error_status"] == 429:
                self.halt("plan limit rejected a request")
            elif rl.get("utilization") is not None and rl["utilization"] >= self.a.stop_util:
                self.halt(f"limit use {rl['utilization']:.2f} reached --stop-util {self.a.stop_util}")
            elif fails >= 3:
                self.halt("three failures in a row")
            if state in ("ok", "bad_fields"):
                break
        return row

    def main(self):
        a = self.a
        os.makedirs(self.outdir, exist_ok=True)
        todo = [i for i in self.batches if check_output(self.out(i), [os.path.basename(n) for n in self.files(i)])[0] not in ("ok", "bad_fields")]
        print(f"[runner] task {self.task}\n[runner] {len(self.batches)} batches requested, {len(self.batches) - len(todo)} already have output, "
              f"{len(todo)} to judge; model {a.model}, {a.jobs} at once; log {self.log_path}", flush=True)
        if a.dry_run:
            print("[runner] dry run, to judge:", todo); return 0
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            self.rows = list(ex.map(self.run_batch, self.batches))
        c = {}
        for r in self.rows: c[r["state"]] = c.get(r["state"], 0) + 1
        done = [r for r in self.rows if "usage" in r]
        tok = {k: sum(r["usage"][k] for r in done) for k in ("cache_write", "cache_read", "out", "requests", "requests_over_100k")}
        cost = sum(r["cost_usd_list"] or 0 for r in done)
        flags = [r["batch"] for r in done if r["audit"]["short"] or r["audit"]["filter_cmds"]]
        models = sorted({str(r["model"]) for r in done})
        print(f"[runner] SUMMARY states {c}; judged now {len(done)} on {models}; tokens {tok}; list-price cost ${cost:.2f}; "
              f"limit use first/last {self.util[:1]}/{self.util[-1:]}; read-audit flags {flags}; "
              f"stopped early: {self.stop_reason or 'no'}", flush=True)
        bad = [r["batch"] for r in self.rows if r["state"] not in ("ok", "skipped_existing")]
        if bad: print(f"[runner] not ok: {bad}", flush=True)
        return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True); ap.add_argument("--batches", required=True)
    ap.add_argument("--model", default="sonnet", help="judge model alias passed to the CLI (default sonnet, the agent definition's model)")
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--retries", type=int, default=1); ap.add_argument("--stop-util", type=float, default=0.95)
    ap.add_argument("--log"); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.jobs <= MAX_JOBS: ap.error(f"--jobs must be 1..{MAX_JOBS}")
    sys.exit(Runner(args).main())
