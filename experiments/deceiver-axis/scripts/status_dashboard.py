#!/usr/bin/env python
"""Build one self-contained HTML status page for the schemer-axis study.

The page is a static file (no server, nothing listening on the cluster): download status.html and open it in a
browser, or open status.ipynb in VS Code, which holds the same page as the saved output of one cell.
The page has five tabs (all sections stay visible, one under the other, where scripts do not run):
  * Overview: stat tiles (tasks running and queued, work in progress, runs recorded, charts), the in-progress
    items with file-based progress, the user's Slurm queue (one squeue call), the six latest results, and a
    model x experiment matrix built from the "### <title> — <date>: <headline>" entries of FINDINGS.md (every
    analysed run is recorded there, so the page follows the log with no second registry),
  * In progress: one card per item of dashboard/in_progress.json and the full queue table,
  * Results: every FINDINGS entry as a card grouped by day (headline, full text, the charts it names), filtered
    by model, experiment kind, text search or a click on a matrix cell,
  * Charts: every indexed chart in a gallery split into groups (main gi_v2 results, Goodfire axis, controls,
    capability retention, interpretability, ..., earlier versions last; rules in CHART_GROUPS / CHART_RULES),
    filtered by model or group, with a viewer (previous / next, the `code <path>` command),
  * Status log: the dated paragraphs of the "Current status" section as a collapsible list.
Light and dark themes (button in the header); colours are the reference palette of the dataviz skill.
Under a VS Code notebook the editor scrolls the output from outside, so there the page never scrolls itself and
the chart viewer opens in place instead of as a viewport overlay.

Usage (repo root):
  OPENSSL_CONF=/dev/null .venv-313/bin/python experiments/deceiver-axis/scripts/status_dashboard.py
      [--out experiments/deceiver-axis/dashboard/status.html] [--rescan-charts] [--no-squeue]

Charts: the list of PNG files is cached in dashboard/charts_index.txt (a bounded find under the run tree and
post/figures); --rescan-charts rebuilds it, and a chart path named in in_progress.json or passed with
--chart is added without a scan. Images are shrunk (width <= 1700 px, WebP quality 85) before embedding; the originals stay where they are and
each figure caption carries its `code <path>` command.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import html
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
RUN_ROOT = Path(os.path.expanduser("~/nobackup/autodelete/self-concept-experiments/schemer-axis"))

MODELS = [
    ("gptoss", "gpt-oss-120b"),
    ("muse", "Muse-Glimmer-30B"),
    ("gemma31", "Gemma 4 31B"),
    ("qwen", "Qwen3.8-27B"),
    ("gemma12", "Gemma 4 12B"),
    ("methods", "Methods / cross-model"),
]
MODEL_LABEL = dict(MODELS)

# first matching rule wins; matched against the entry title
CATEGORIES = [
    ("gfar", "Goodfire RH axis (gfar)", r"Goodfire|gfar"),
    ("floor", "Floor (inverse cap)", r"\bfloor\b"),
    ("pct", "Control-direction percentile", r"antonym|covariance-shaped|Percentile design|Control directions"),
    ("random", "Random-direction controls", r"\brandom\b|noise control|\b1729\b|\b1730\b"),
    ("cap", "Cap", r"\bcap\b|Capping|cap thresholds"),
    ("pool", "Task pool and calibration", r"problem-pool|data expansion|LiveCodeBench|hidden-pass audit|Unsteered hard45|band many-seed|capability check"),
    ("capability", "Capability retention", r"capability|benchmarks|Capability"),
    ("probes", "Probes and anecdotes", r"probes?\b|anecdote|neutral-question"),
    ("prompt", "Prompt interventions", r"honesty-prompt|prompt-rh"),
    ("harness", "Harness and smoke tests", r"smoke|backend|Tool check|Concurrent episodes|token budget"),
    ("direction", "Direction extraction and geometry", r"[Ee]xtraction|contrast|vector: validation|axes for|Acting|Phase 1|Phase 2|lens|Gemma 4 pilot$"),
    ("steer", "Steering dose response", r"dose|main run|pilot arm|arm, judged|gi_v2 run|half-dose|judged outcome|Main-run|main-run|layer pilot|negative-dose|vector pilot"),
    ("other", "Other", r""),
]
CAT_ORDER = ["steer", "cap", "floor", "random", "pct", "capability", "probes", "prompt", "gfar", "direction", "pool", "harness", "other"]
CAT_LABEL = {k: lab for k, lab, _ in CATEGORIES}

# Charts tab: every chart goes into one group, shown as its own section in this order.
CHART_GROUPS = [
    ("main", "Main results: gi_v2 steering, cap and floor",
     "The current judged chart per model for the gi_v2 steering dose response, the cap ladder and the floor ladder."),
    ("gfar", "Goodfire RH axis (gfar)", "The Goodfire reward-hacking axis: comparison with gi_v2, steering and capping along it."),
    ("controls", "Random-direction and other controls", "Random-direction arms, the 1729 / 1730 follow-ups and the control sets."),
    ("capability", "Capability retention", "Benchmark scores under steering: how much capability each dose or control keeps."),
    ("interp", "Interpretability: lens and direction readouts", "Logit-lens and hack-direction readouts of what a steering vector changes downstream."),
    ("prompt", "Prompt interventions", "Honesty-prompt arms, judged like the steering arms."),
    ("methods", "Pipeline and method figures", "Diagrams of the extraction and steering pipeline."),
    ("other", "Other", "Charts no rule places yet (add a rule to CHART_RULES in scripts/status_dashboard.py)."),
    ("earlier", "Earlier versions and earlier vector sets",
     "Charts replaced by a newer version above (fewer arms or seeds), the first gi vector set (17 roles), and the October post figures."),
]
# First matching rule wins; the regex is searched in the full chart path. A chart with no match takes the group of the
# category of the result that names it (CAT_TO_GROUP), else "other". Before the rules, a chart in an analysis_<name><N>
# folder goes to "earlier" when a sibling folder with the same name and a higher N holds a chart (analysis_all5 / all7 /
# all8: only all8 stays current).
CHART_RULES = [
    ("earlier", r"/gi/pilot/|muse_glimmer/capeval/|/post/figures/|hard108/gemma_hard108_judged_chart\.png|gi_v2/pilot/gi_main_chart\.png"
                r"|qwen_hard61/analysis_main/"),
    ("methods", r"pipeline_figure"),
    ("gfar", r"gfar|[Gg]oodfire|/gf_|gfscan|gfcap"),
    ("capability", r"capeval"),
    ("interp", r"lens|randscreen|separation"),
    ("prompt", r"prompt_rh"),
    ("controls", r"rand|ctl|sign_pair|problem_seed|1729|1730|1731|antonym"),
    ("main", r"capladder|floorladder|_floor|judged_chart|main_chart|vs_17role"),
]
CAT_TO_GROUP = {"steer": "main", "cap": "main", "floor": "main", "random": "controls", "pct": "controls", "capability": "capability",
                "probes": "interp", "direction": "interp", "prompt": "prompt", "gfar": "gfar"}
VERSIONED_RE = re.compile(r"^(.*/analysis_[A-Za-z_]*?)(\d+)/[^/]+$")


def chart_groups(paths: list[str], chart_entry: dict) -> dict[str, str]:
    newest: dict[str, int] = {}
    for p in paths:
        m = VERSIONED_RE.match(p)
        if m:
            newest[m.group(1)] = max(newest.get(m.group(1), -1), int(m.group(2)))
    out = {}
    for p in paths:
        m = VERSIONED_RE.match(p)
        if m and int(m.group(2)) < newest[m.group(1)]:
            out[p] = "earlier"
            continue
        g = next((k for k, rx in CHART_RULES if re.search(rx, p)), None)
        if g is None:
            e = chart_entry.get(p)
            g = CAT_TO_GROUP.get(e["cat"], "other") if e else "other"
        out[p] = g
    return out


MAIN_KIND = ["Steering dose response", "Cap ladder", "Floor ladder"]


def main_group_key(path: str) -> tuple[int, int]:
    """Order inside the main group: steering dose response, then cap ladder, then floor ladder; models in page order."""
    kind = 1 if "capladder" in path else 2 if "floor" in path else 0
    order = [m for m, _ in MODELS]
    return kind, order.index(chart_model(path))

HEAD_RE = re.compile(r"^### (?P<title>.*?) — (?P<date>20\d\d-\d\d-\d\d(?: to [\d-]+)?)\s*:\s*(?P<headline>.*)$")
PNG_RE = re.compile(r"[A-Za-z0-9_./+-]+\.png")


def classify_models(title: str, headline: str, date: str) -> list[str]:
    def hits(t: str) -> list[str]:
        out = []
        if re.search(r"gpt-oss|GPT-OSS", t):
            out.append("gptoss")
        if re.search(r"\bMuse\b", t):
            out.append("muse")
        if "Gemma" in t:
            has12, has31 = "12B" in t, "31B" in t
            if has31:
                out.append("gemma31")
            if has12 or (not has31 and re.search(r"Gemma 4 (pilot|Phase 3|problem-pool)", t)):
                out.append("gemma12")
            if not has12 and not has31 and "gemma12" not in out:
                out.append("gemma31")
        if "Qwen" in t:
            out.append("qwen")
        return out

    found = hits(title)
    if not found:
        found = hits(headline[:700])
    if not found and date[:10] <= "2026-10-13":
        found = ["gptoss"]  # the first-study entries carry no model name; they are all gpt-oss-120b
    return found or ["methods"]


def classify_category(title: str) -> str:
    for key, _lab, rx in CATEGORIES:
        if rx and re.search(rx, title):
            return key
    return "other"


def first_sentences(text: str, limit: int = 380) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.end() for m in re.finditer(r"[.;] (?=[A-Z`(*])", cut) if m.end() > 120]
    if ends:
        return cut[: ends[-1]].rstrip(" ;")
    return cut.rsplit(" ", 1)[0] + " …"


def parse_findings(path: Path):
    lines = path.read_text().splitlines()
    status_md, entries = [], []
    section = None
    cur = None
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip()
            if cur:
                entries.append(cur)
                cur = None
            continue
        if section == "Current status":
            status_md.append(line)
        elif section == "Runs":
            if line.startswith("### "):
                if cur:
                    entries.append(cur)
                m = HEAD_RE.match(line)
                if m:
                    cur = {"title": m["title"].strip(), "date": m["date"], "headline": m["headline"].strip(), "body": []}
                else:
                    cur = {"title": line[4:].strip(), "date": "", "headline": "", "body": []}
            elif cur is not None:
                cur["body"].append(line)
    if cur:
        entries.append(cur)
    for i, e in enumerate(entries):
        e["id"] = f"e{i}"
        e["order"] = i
        e["models"] = classify_models(e["title"], e["headline"], e["date"])
        e["cat"] = classify_category(e["title"])
        e["body_md"] = "\n".join(e.pop("body")).strip()
        e["summary"] = first_sentences(e["headline"])
    return "\n".join(status_md).strip(), entries


# ---------------------------------------------------------------- charts
def chart_index(cache: Path, rescan: bool, extra: list[str]) -> list[str]:
    paths: list[str] = []
    if cache.exists() and not rescan:
        paths = [p for p in cache.read_text().splitlines() if p.strip()]
    else:
        cmd = ["find", str(RUN_ROOT), "-maxdepth", "6", "-name", "*.png", "-not", "-path", "*/dossiers/*", "-not", "-path", "*/archive_*/*", "-not", "-name", "draft_*"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
            paths = [p for p in out.splitlines() if p.strip()]
        except Exception as exc:  # noqa: BLE001
            print(f"chart scan failed: {exc}", file=sys.stderr)
        paths += sorted(str(p) for p in (EXP / "post" / "figures").glob("*.png"))
    for p in extra:
        if p not in paths:
            paths.append(p)
    paths = sorted({p for p in paths if os.path.isfile(p)})
    cache.write_text("\n".join(paths) + "\n")
    return paths


MODEL_PATH_HINT = {
    "muse": ("muse_glimmer",),
    "gemma31": ("gemma4_31b",),
    "gemma12": ("gemma4_12b",),
    "qwen": ("qwen38",),
    "gptoss": ("/gi_v2/", "/gi/", "/main", "/capeval/", "/pilot/", "prompt_rh"),
}


def chart_model(path: str) -> str:
    rel = path.replace(str(RUN_ROOT), "")
    for key in ("muse", "gemma31", "gemma12", "qwen"):
        if any(h in rel for h in MODEL_PATH_HINT[key]):
            return key
    if any(rel.startswith(h) or h in rel for h in MODEL_PATH_HINT["gptoss"]):
        return "gptoss"
    return "methods"


def resolve_chart(ref: str, charts: list[str], models: list[str]) -> str | None:
    ref = ref.lstrip("./")
    cands = [p for p in charts if p.endswith("/" + ref)]
    if not cands:
        base = ref.rsplit("/", 1)[-1]
        cands = [p for p in charts if p.rsplit("/", 1)[-1] == base]
    if not cands:
        return None
    if len(cands) > 1:
        hinted = [p for p in cands if chart_model(p) in models]
        cands = hinted or cands
    return max(cands, key=lambda p: (-(p.count("/")), os.path.getmtime(p))) if len(cands) > 1 else cands[0]


def embed_png(path: str, max_w: int = 1700) -> tuple[str, int, int]:
    from PIL import Image

    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (252, 252, 251, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=85, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(), im.width, im.height


# ---------------------------------------------------------------- queue
def expand_count(spec: str) -> int:
    """Number of tasks in a pending array id such as 123_[0-3], 123_[2-3%2], 123_[1,4,7-9]."""
    m = re.search(r"_\[(.*?)\]", spec)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split("%")[0].split(","):
        if "-" in part:
            a, b = part.split("-")[:2]
            n += int(b.split(":")[0]) - int(a) + 1
        elif part:
            n += 1
    return n


def elapsed_seconds(s: str) -> int:
    """Slurm elapsed string ([D-]H:M:S, M:S) in seconds; 0 for an empty string."""
    if not s:
        return 0
    days, _, rest = s.rpartition("-")
    n = 0
    for part in rest.split(":"):
        n = n * 60 + int(part)
    return n + 86400 * int(days or 0)


QUEUE_CACHE = Path.home() / "nobackup/autodelete/self-concept-experiments/schemer-axis/gfar_vs_gi_v2/.squeue_cache"


def read_queue(skip: bool):
    if skip:
        return None, "queue not read (--no-squeue)"
    fmt = "%i|%F|%P|%q|%j|%T|%M|%l|%S|%R|%b"
    # A running job monitor already polls the scheduler every 120 s and leaves the listing (same format) in QUEUE_CACHE;
    # use it while it is fresh so a rebuild adds no scheduler call (ORC rule: status checks at least 60 s apart).
    if QUEUE_CACHE.is_file() and time.time() - QUEUE_CACHE.stat().st_mtime < 150:
        listing = QUEUE_CACHE.read_text()
    else:
        try:
            r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", fmt],
                               capture_output=True, text=True, timeout=45)
        except Exception as exc:  # noqa: BLE001
            return None, f"squeue failed: {exc}"
        if r.returncode != 0:
            return None, f"squeue failed: {r.stderr.strip()[:200]}"
        listing = r.stdout
    groups: dict[str, dict] = {}
    for line in listing.splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) < 11:
            continue
        jid, arr, part, qos, name, state, elapsed, limit, start, reason, gres = f[:11]
        g = groups.setdefault(arr, {"id": arr, "name": name, "parts": set(), "qos": qos, "limit": limit,
                                    "running": 0, "pending": 0, "other": {}, "elapsed": "", "reasons": set(),
                                    "nodes": set(), "start": "", "gres": gres})
        g["parts"].update(part.split(","))
        if state == "RUNNING":
            g["running"] += 1
            g["nodes"].add(reason)
            if elapsed_seconds(elapsed) > elapsed_seconds(g["elapsed"]):
                g["elapsed"] = elapsed
        elif state == "PENDING":
            g["pending"] += expand_count(jid)
            g["reasons"].add(reason.strip("()"))
            if start and start != "N/A":
                g["start"] = start
        else:
            g["other"][state] = g["other"].get(state, 0) + 1
    return list(groups.values()), ""


def fmt_start(s: str) -> str:
    try:
        t = dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return ""
    return t.strftime("%b %-d %-I:%M %p")


# ---------------------------------------------------------------- in-progress items
def progress_of(spec: dict | None):
    if not spec:
        return None
    files = sorted(glob.glob(os.path.expanduser(os.path.expandvars(spec["glob"]))))
    if spec.get("kind") == "lines":
        done = 0
        for p in files:
            try:
                with open(p, "rb") as fh:
                    done += sum(1 for _ in fh)
            except OSError:
                pass
    else:
        done = len(files)
    return done, int(spec.get("total", 0)), spec.get("unit", "")


def last_line(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(os.path.expanduser(os.path.expandvars(path))) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def item_state(item: dict, queue) -> str:
    if item.get("state"):
        return item["state"]
    ids = {str(j) for j in item.get("jobs", [])}
    if queue is None or not ids:
        return "in progress"
    mine = [g for g in queue if g["id"] in ids]
    if any(g["running"] for g in mine):
        return "running"
    if mine:
        return "queued"
    return "jobs done, analysis pending"


# ---------------------------------------------------------------- html
MODEL_SHORT = {"gptoss": "gpt-oss", "muse": "Muse", "gemma31": "Gemma 31B", "qwen": "Qwen", "gemma12": "Gemma 12B", "methods": "Methods"}
ACTIVE_STATES = ["running", "queued", "jobs done, analysis pending", "in progress"]
STATE_STYLE = {
    "running": ("b-run", "●", "running"),
    "queued": ("b-queue", "○", "queued"),
    "jobs done, analysis pending": ("b-pend", "◐", "analysis pending"),
    "in progress": ("b-run", "●", "in progress"),
    "interim result": ("b-plain", "◆", "interim result"),
    "done": ("b-done", "✓", "done"),
}

# Colours: the reference palette of the dataviz skill (chrome, ink, categorical slots 1-5 for the five models in a
# fixed order, both modes checked with validate_palette.js). A model dot always sits beside the model's name.
CSS = """
body.dash-page{margin:0;background:#f9f9f7}
body.dash-page:has(.dash[data-theme=dark]){background:#0d0d0d}
body.dash-page .dash{min-height:100vh}
.dash{
  --plane:#f9f9f7;--card:#fcfcfb;--ink:#141413;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--line2:#c3c2b7;--soft:#f1f0ec;--paper:#fcfcfb;
  --accent:#2a78d6;--accent-ink:#1c5cab;--track:#dce8f8;--amber:#eda100;
  --run-bg:#e2edfb;--run-fg:#184f95;--q-bg:#fcefd0;--q-fg:#7a5200;--pend-bg:#eceafb;--pend-fg:#4034a0;--ok-bg:#e1f2e1;--ok-fg:#006300;
  --m-gptoss:#2a78d6;--m-muse:#eb6834;--m-gemma31:#1baf7a;--m-qwen:#eda100;--m-gemma12:#e87ba4;--m-methods:#898781;
  --t1:6%;--t2:11%;--t3:17%;--t4:24%;--t5:32%;
  --shadow:0 1px 2px rgba(11,11,11,.04),0 2px 10px rgba(11,11,11,.03);
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  color-scheme:light;background:var(--plane);color:var(--ink);text-align:left;
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  padding-bottom:56px;
  *{box-sizing:border-box}
  [hidden]{display:none!important}
  table{border-collapse:collapse;border-spacing:0;font-size:inherit}
  th,td{text-align:left;border:0;background:none;color:inherit;font-size:inherit}
  table tr,tbody tr:nth-child(n),tbody tr:hover{background:none}
  a{color:var(--accent-ink)}
  .in{max-width:1360px;margin:0 auto;padding:0 28px}
  .mono{font-family:var(--mono);font-size:12px}
  .small{font-size:12.5px;color:var(--ink2)}
  code{font-family:var(--mono);font-size:.88em;background:var(--soft);color:var(--ink);padding:1px 4px;border-radius:4px}
  button{font:inherit}
  :focus-visible{outline:2px solid var(--accent);outline-offset:2px}

  /* header */
  header.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding-top:20px;padding-bottom:14px}
  .logo{width:36px;height:36px;border-radius:10px;background:var(--ink);color:var(--plane);display:grid;place-items:center;flex:none}
  .logo svg{width:20px;height:20px}
  h1{font-size:19px;line-height:1.2;margin:0;font-weight:650;letter-spacing:-.01em;color:var(--ink)}
  .meta{color:var(--ink2);font-size:12.5px;margin-top:2px}
  .snap{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink2)}
  .stamp{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);background:var(--card);border-radius:999px;padding:4px 12px;white-space:nowrap}
  .stamp b{color:var(--ink);font-weight:600}
  .stamp .sd{width:7px;height:7px;border-radius:50%;background:var(--ok-fg)}
  .stamp.stale{border-color:var(--amber);background:var(--q-bg);color:var(--q-fg)}
  .stamp.stale b{color:var(--q-fg)}
  .stamp.stale .sd{background:var(--amber)}
  .iconbtn{appearance:none;border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:999px;width:30px;height:30px;cursor:pointer;display:grid;place-items:center;font-size:14px;padding:0}
  .iconbtn:hover{color:var(--ink);border-color:var(--line2)}

  /* tabs */
  nav.tabs{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--plane) 90%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
  nav.tabs .in{display:flex;gap:2px;overflow-x:auto}
  .tab{appearance:none;border:0;background:none;font-weight:550;color:var(--ink2);padding:11px 14px 10px;border-bottom:2px solid transparent;margin-bottom:-1px;cursor:pointer;display:flex;gap:8px;align-items:center;white-space:nowrap}
  .tab:hover{color:var(--ink)}
  .tab.on{color:var(--ink);border-bottom-color:var(--accent)}
  .tab .ct{font-size:11.5px;font-weight:600;background:var(--soft);color:var(--ink2);border-radius:999px;padding:0 7px;line-height:18px;font-variant-numeric:tabular-nums}
  .tab.on .ct{background:var(--run-bg);color:var(--run-fg)}
  .panel{padding-top:4px}
  h2.ph{font-size:17px;margin:30px 0 4px;padding-bottom:6px;border-bottom:1px solid var(--line)}
  h2.sec{font-size:14.5px;font-weight:650;margin:26px 0 10px;display:flex;align-items:baseline;gap:8px}
  h2.sec .n{font-weight:500;color:var(--ink2);font-size:12.5px}

  /* shared pieces */
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}
  .card>.ch{display:flex;align-items:baseline;gap:10px;padding:14px 18px 0}
  .card>.ch h2{font-size:14.5px;margin:0;font-weight:650;color:var(--ink)}
  .card>.ch .hint{color:var(--ink2);font-size:12.5px}
  .card>.cb{padding:6px 18px 14px}
  .link{appearance:none;border:0;background:none;padding:0;color:var(--accent-ink);cursor:pointer;font-size:12.5px;font-weight:550;margin-left:auto;white-space:nowrap}
  .link:hover{text-decoration:underline}
  .badge{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;white-space:nowrap;line-height:19px}
  .badge .g{font-size:10px;line-height:1}
  .b-run{background:var(--run-bg);color:var(--run-fg)}
  .b-queue{background:var(--q-bg);color:var(--q-fg)}
  .b-pend{background:var(--pend-bg);color:var(--pend-fg)}
  .b-done{background:var(--ok-bg);color:var(--ok-fg)}
  .b-plain{background:var(--soft);color:var(--ink2)}
  .mdot{display:inline-block;width:9px;height:9px;border-radius:50%;flex:none;background:var(--m-methods)}
  .m-gptoss{background:var(--m-gptoss)}.m-muse{background:var(--m-muse)}.m-gemma31{background:var(--m-gemma31)}
  .m-qwen{background:var(--m-qwen)}.m-gemma12{background:var(--m-gemma12)}.m-methods{background:var(--m-methods)}
  .mtag{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
  .chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;border:1px solid var(--line);border-radius:999px;padding:0 9px;line-height:20px;color:var(--ink2);background:var(--card);white-space:nowrap}
  .chip.cat{background:var(--soft);border-color:transparent}
  .prog .pl{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
  .prog .pl span{white-space:nowrap}
  .prog .pl b{color:var(--ink);font-weight:600}
  .prog .pp{font-weight:600;color:var(--ink)}
  .bar{height:8px;background:var(--track);border-radius:4px;overflow:hidden;margin-top:5px}
  .bar>i{display:block;height:100%;background:var(--accent);border-radius:4px}

  /* overview */
  .kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:20px 0 16px}
  .kpi{appearance:none;text-align:left;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 16px 12px;box-shadow:var(--shadow);cursor:pointer;min-width:0}
  .kpi:hover{border-color:var(--line2)}
  .kpi .l{font-size:12.5px;color:var(--ink2);display:flex;align-items:center;gap:6px}
  .kpi .kg{font-size:10px;line-height:1}
  .kpi .kg.run{color:var(--accent)}
  .kpi .kg.q{color:var(--amber)}
  .kpi .v{font-size:30px;font-weight:650;line-height:1.2;letter-spacing:-.02em;margin-top:1px;color:var(--ink)}
  .kpi .v small{font-size:13px;font-weight:500;color:var(--ink2);letter-spacing:0;margin-left:6px}
  .kpi .s{font-size:12.5px;color:var(--ink2);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .grid2{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(0,1fr);gap:16px;margin-bottom:16px;align-items:start}
  .wrow{display:grid;grid-template-columns:minmax(0,1fr) 176px 132px;gap:14px;padding:11px 0;border-top:1px solid var(--line);align-items:center}
  .wrow:first-child{border-top:0}
  .wrow .t{font-weight:600;line-height:1.35}
  .wrow .u{font-size:12.5px;color:var(--ink2);margin-top:3px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .wrow .st{justify-self:end}
  .stack{display:flex;flex-direction:column;gap:16px;min-width:0}
  .qrow{display:grid;grid-template-columns:minmax(0,1fr) 176px 132px;gap:14px;padding:10px 0;border-top:1px solid var(--line);align-items:center}
  .qrow:first-child{border-top:0}
  .qrow .t{font-weight:600;line-height:1.35}
  .qrow .u{font-size:12.5px;color:var(--ink2)}
  .qrow .seg{width:auto;margin:0 0 4px}
  .qrow .el{justify-self:end;text-align:right;font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
  a.lrow{display:grid;grid-template-columns:50px minmax(0,1fr);gap:2px 12px;padding:10px 0;border-top:1px solid var(--line);color:inherit;text-decoration:none}
  a.lrow:first-child{border-top:0}
  .lrow .d{font-size:12.5px;color:var(--ink2);white-space:nowrap;padding-top:1px}
  .lrow .t{font-weight:600;line-height:1.35}
  a.lrow:hover .t{color:var(--accent-ink)}
  .lrow .s{grid-column:2;color:var(--ink2);font-size:13px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .lrow .m{grid-column:2;display:flex;gap:10px;font-size:12px;color:var(--ink2);margin-top:2px}

  /* matrix */
  .mwrap{overflow-x:auto;padding:4px 14px 6px}
  table.matrix{border-collapse:separate;border-spacing:4px;width:100%;min-width:900px;table-layout:fixed}
  .matrix th{font-weight:600;font-size:12.5px;text-align:left;padding:4px 8px;color:var(--ink)}
  .matrix thead th{vertical-align:bottom;padding-bottom:6px}
  .matrix th.mh,.matrix th.rh{cursor:pointer;border-radius:8px}
  .matrix th.mh:hover,.matrix th.rh:hover{background:var(--soft)}
  .matrix th.mh .n{display:block;font-weight:500;color:var(--ink2);font-size:12px;margin-top:1px}
  .matrix th.rh{width:222px;font-weight:550}
  .matrix td{border-radius:8px;padding:6px 10px;height:44px;vertical-align:middle;font-variant-numeric:tabular-nums}
  .matrix td.empty{color:var(--line2);text-align:left}
  .matrix td.cell{cursor:pointer;background:var(--soft)}
  .matrix td.t1{background:color-mix(in srgb,var(--accent) var(--t1),var(--card))}
  .matrix td.t2{background:color-mix(in srgb,var(--accent) var(--t2),var(--card))}
  .matrix td.t3{background:color-mix(in srgb,var(--accent) var(--t3),var(--card))}
  .matrix td.t4{background:color-mix(in srgb,var(--accent) var(--t4),var(--card))}
  .matrix td.t5{background:color-mix(in srgb,var(--accent) var(--t5),var(--card))}
  .matrix td.cell:hover,.matrix td.sel{outline:2px solid var(--accent);outline-offset:-2px}
  .matrix .n{font-weight:650;font-size:15px;color:var(--ink)}
  .matrix .d{font-size:11.5px;color:var(--ink2);margin-left:6px}
  .matrix .w{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:2px}
  .matrix .w .d{margin-left:0}
  .legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:12px;color:var(--ink2);padding:4px 18px 14px}
  .legend .ramp{display:inline-flex;gap:2px;align-items:center;margin:0 4px}
  .legend .ramp i{display:block;width:18px;height:10px;border-radius:3px}

  /* in progress */
  .note{background:var(--soft);border-radius:10px;padding:10px 14px;color:var(--ink2);font-size:13px;margin:18px 0 0}
  .note p{margin:0;max-width:130ch}
  .wcard{padding:15px 18px;margin-bottom:12px;display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:8px 30px}
  .wcard .hd{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .wcard h3{font-size:15px;margin:0;font-weight:650;color:var(--ink)}
  .wcard .det{color:var(--ink2);font-size:13px;margin:8px 0 0;max-width:100ch}
  .wcard .side{display:flex;flex-direction:column;gap:10px;border-left:1px solid var(--line);padding-left:22px}
  .kv{display:grid;grid-template-columns:66px minmax(0,1fr);gap:4px 10px;font-size:12.5px;margin:0}
  .kv dt{color:var(--ink2)}
  .kv dd{margin:0;color:var(--ink);overflow-wrap:anywhere}
  .tblwrap{overflow-x:auto}
  table.tbl{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  .tbl th{font-size:12px;font-weight:600;color:var(--ink2);text-align:left;padding:10px 14px;border-bottom:1px solid var(--line);white-space:nowrap}
  .tbl td{padding:11px 14px;border-bottom:1px solid var(--line);vertical-align:top}
  .tbl tr:last-child td{border-bottom:0}
  .tbl .jn{color:var(--ink2);font-size:12.5px}
  .seg{display:flex;gap:2px;height:8px;width:160px;margin:5px 0 5px}
  .seg i{display:block;border-radius:4px;min-width:5px}
  .seg .r{background:var(--accent)}
  .seg .p{background:var(--amber)}
  .meter{height:4px;width:150px;background:var(--track);border-radius:2px;overflow:hidden;margin-top:6px}
  .meter i{display:block;height:100%;background:var(--accent);border-radius:2px}

  /* results */
  .filters{position:sticky;top:44px;z-index:10;background:color-mix(in srgb,var(--plane) 93%,transparent);backdrop-filter:blur(8px);padding:14px 0 10px;border-bottom:1px solid var(--line)}
  .pills{display:flex;gap:6px;flex-wrap:wrap}
  .pill{appearance:none;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:999px;padding:3px 12px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;line-height:20px}
  .pill:hover{border-color:var(--line2);color:var(--ink)}
  .pill .c{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .pill.on{background:var(--ink);color:var(--plane);border-color:var(--ink)}
  .pill.on .c{color:inherit;opacity:.72}
  .frow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:9px}
  .frow select,.frow input[type=search]{font:inherit;font-size:13px;padding:5px 10px;border:1px solid var(--line2);border-radius:8px;background:var(--card);color:var(--ink);height:32px}
  .frow input[type=search]{min-width:260px;flex:1;max-width:520px}
  .frow label{display:inline-flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2);cursor:pointer}
  .frow .btn{appearance:none;font-size:13px;padding:0 12px;height:32px;border:1px solid var(--line2);border-radius:8px;background:var(--card);color:var(--ink);cursor:pointer}
  .frow .count{margin-left:auto;font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
  .day{margin-top:20px}
  .dayh{font-size:12.5px;font-weight:600;color:var(--ink2);margin:0 0 9px;display:flex;gap:10px;align-items:center}
  .dayh::after{content:"";flex:1;height:1px;background:var(--line)}
  .dayh .n{font-weight:500}
  .entry{padding:14px 18px;margin-bottom:10px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:0 20px;scroll-margin-top:140px;transition:box-shadow .3s,border-color .3s}
  .entry.flash{border-color:var(--accent);box-shadow:0 0 0 3px var(--run-bg)}
  .entry h3{font-size:14.5px;margin:0 0 6px;font-weight:650;line-height:1.35;color:var(--ink)}
  .entry .sum{margin:8px 0 8px;max-width:108ch}
  .entry .thumb{grid-column:2;grid-row:1;width:156px;height:100px;border:1px solid var(--line);border-radius:8px;background:var(--paper);cursor:zoom-in;padding:4px;display:grid;place-items:center;appearance:none}
  .entry .thumb img{max-width:100%;max-height:90px;width:auto;height:auto;display:block}
  .entry details{grid-column:1 / -1}
  details>summary{cursor:pointer;color:var(--accent-ink);font-size:12.5px;font-weight:550;user-select:none}
  .body{margin-top:10px;border-top:1px solid var(--line);padding-top:10px;overflow-x:auto;font-size:13.5px;line-height:1.6}
  .body p{max-width:108ch;margin:8px 0}
  .body li{max-width:104ch}
  .body table{border-collapse:collapse;font-size:12.5px;margin:10px 0;font-variant-numeric:tabular-nums}
  .body th,.body td{border:1px solid var(--line);padding:4px 9px;text-align:left;vertical-align:top}
  .body th{background:var(--soft);font-weight:600}
  .figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;margin:2px 0 12px}
  .figs figure{margin:0;border:1px solid var(--line);border-radius:10px;padding:8px;background:var(--paper);color:#52514e;min-width:0;max-width:980px}
  .figs img{max-width:100%;height:auto;display:block;cursor:zoom-in;margin:0 auto}
  .figs figcaption{font-size:11.5px;margin-top:6px;word-break:break-all}
  .figs figcaption code{background:#f1f0ec;color:#141413;user-select:all}
  .empty-note{color:var(--ink2);padding:28px 0;text-align:center}

  /* charts */
  .gbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:14px 0 0}
  .gbar label{display:inline-flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2);cursor:pointer}
  .gbar .count{margin-left:auto;font-size:12.5px;color:var(--ink2)}
  .gbar2{margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}  /* faint rule: the model pills and the group pills are two separate filters */
  .gsec{margin-top:26px;scroll-margin-top:140px}
  .gsech{font-size:15px;font-weight:650;color:var(--ink);margin:0;display:flex;gap:10px;align-items:baseline}
  .gsech::after{content:"";flex:1;height:1px;background:var(--line);align-self:center}
  .gsech .n{font-size:12.5px;font-weight:500;color:var(--ink2);font-variant-numeric:tabular-nums}
  .gdesc{margin:4px 0 0;font-size:13px;color:var(--ink2);max-width:92ch}
  .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:14px}
  .tile{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column;min-width:0}
  .tile .ph{height:206px;background:var(--paper);display:grid;place-items:center;border-bottom:1px solid var(--line);cursor:zoom-in;padding:8px}
  .tile .ph img{max-width:100%;max-height:190px;width:auto;height:auto;display:block}
  .tile figcaption{padding:10px 13px 11px;min-width:0}
  .tile .tt{font-weight:600;font-size:13px;line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}
  .tile .ts{font-size:12px;color:var(--ink2);margin-top:5px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .tile .ts .link{margin-left:auto;font-size:12px}

  /* status log */
  .loglist{overflow:hidden}
  .logi{border-top:1px solid var(--line)}
  .logi:first-child{border-top:0}
  .logi[open]{background:color-mix(in srgb,var(--soft) 45%,var(--card))}
  .logi>summary{list-style:none;display:grid;grid-template-columns:64px minmax(0,1fr) 14px;gap:12px;padding:9px 18px;color:var(--ink);font-size:13.5px;font-weight:550;align-items:baseline}
  .logi>summary::-webkit-details-marker{display:none}
  .logi>summary .d{color:var(--ink2);font-size:12.5px;font-weight:500;white-space:nowrap}
  .logi>summary::after{content:"+";color:var(--ink2);font-weight:500}
  .logi[open]>summary::after{content:"−"}
  .logi>summary:hover{background:var(--soft)}
  .logi .body{margin:0;padding:10px 18px 14px 94px;border-top:0}
  .logrest{padding:12px 18px}

  .foot{margin-top:40px;font-size:12px;color:var(--ink2);display:flex;gap:16px;flex-wrap:wrap}
  .foot kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line2);border-bottom-width:2px;border-radius:4px;padding:0 4px;background:var(--card);color:var(--ink)}

  /* lightbox */
  .lb{position:fixed;inset:0;z-index:3000;background:rgba(13,13,13,.93);display:none;flex-direction:column;color:#f0efec}
  .lb.on{display:flex}
  .lb button{appearance:none;font-size:13px;color:#f0efec;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);border-radius:8px;padding:5px 12px;cursor:pointer;white-space:nowrap}
  .lb button:hover{background:rgba(255,255,255,.18)}
  .lb-top{display:flex;align-items:center;gap:12px;padding:12px 16px}
  .lb-title{font-weight:600;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .lb-n{font-size:12.5px;color:#c3c2b7;font-variant-numeric:tabular-nums}
  .lb-stage{flex:1;min-height:0;overflow:auto;display:flex;padding:0 60px 4px}
  .lb-stage img{margin:auto;max-width:100%;max-height:100%;width:auto;height:auto;background:#fcfcfb;border-radius:6px;cursor:zoom-in;display:block}
  .lb-stage.zoom{padding:0 12px 4px}
  .lb-stage.zoom img{max-width:none;max-height:none;cursor:zoom-out}
  .lb .lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:40px;height:60px;font-size:22px;padding:0}
  .lb-nav.prev{left:10px}
  .lb-nav.next{right:10px}
  .lb-bot{display:flex;gap:10px;align-items:center;padding:10px 16px 14px;font-size:12.5px;color:#c3c2b7}
  .lb-bot span{white-space:nowrap}
  .lb-bot code{background:rgba(255,255,255,.1);color:#f0efec;user-select:all;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;padding:4px 8px;font-size:12px}

  @media (max-width:1080px){
    .kpis{grid-template-columns:repeat(3,minmax(0,1fr))}
    .matrix .lw{display:none}
    .grid2{grid-template-columns:minmax(0,1fr)}
    .wcard{grid-template-columns:minmax(0,1fr)}
    .wcard .side{border-left:0;padding-left:0;border-top:1px solid var(--line);padding-top:12px}
  }
  @media (max-width:720px){
    .in{padding:0 14px}
    .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
    .wrow{grid-template-columns:minmax(0,1fr)}
    .wrow .st{justify-self:start}
    .entry .thumb{display:none}
    .logi .body{padding-left:18px}
  }
}
.dash{position:relative}
.dash.embed{min-height:800px}
.dash.embed .lb{position:absolute;inset:auto 0 auto 0;height:760px;border-radius:12px}
.dash.js .panel{display:none}
.dash.js .panel.on{display:block}
.dash.js h2.ph{display:none}
.dash:not(.js) nav.tabs{display:none}
.dash[data-theme=dark]{
  --plane:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--line2:#44443f;--soft:#252523;
  --accent:#3987e5;--accent-ink:#86b6ef;--track:#1f3654;--amber:#c98500;
  --run-bg:#172c47;--run-fg:#9ec5f4;--q-bg:#3a2c08;--q-fg:#f2c14e;--pend-bg:#26233f;--pend-fg:#b3abf2;--ok-bg:#12301a;--ok-fg:#5fcf5f;
  --m-gptoss:#3987e5;--m-muse:#d95926;--m-gemma31:#199e70;--m-qwen:#c98500;--m-gemma12:#d55181;--m-methods:#898781;
  --t1:9%;--t2:16%;--t3:24%;--t4:33%;--t5:44%;
  --shadow:none;color-scheme:dark;
}
"""

JS = """
(function(){
const root=document.querySelector('.dash');if(!root)return;root.classList.add('js');
const $=(s,r)=>(r||root).querySelector(s),$$=(s,r)=>Array.from((r||root).querySelectorAll(s));
const store={get(k){try{return localStorage.getItem(k);}catch(e){return null;}},set(k,v){try{localStorage.setItem(k,v);}catch(e){}}};
const IMG=JSON.parse(document.getElementById('dash-imgdata').textContent);
/* A notebook output in VS Code sits in a document the editor scrolls from outside (overflow hidden): there the page must not
   scroll itself, and a viewport-fixed overlay would not line up with what is on screen, so the chart viewer opens in place. */
const managed=!document.body.classList.contains('dash-page')&&[document.documentElement,document.body].some(e=>/hidden|clip/.test(getComputedStyle(e).overflowY));
if(managed)root.classList.add('embed');
function toTop(){if(managed)return;const top=main.getBoundingClientRect().top-nav.offsetHeight;if(top<0)window.scrollBy(0,top);}

/* images are stored once and attached when they come into view */
const io=('IntersectionObserver' in window)?new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const im=e.target;if(!im.getAttribute('src'))im.src=IMG[im.dataset.img];io.unobserve(im);}}),{rootMargin:'400px'}):null;
$$('img[data-img]').forEach(im=>{if(io)io.observe(im);else im.src=IMG[im.dataset.img];});

/* tabs */
const tabs=$$('.tab'),panels=$$('.panel'),nav=$('nav.tabs'),main=$('main');
function go(name,keepScroll){
  if(!panels.some(p=>p.dataset.p===name))name='overview';
  tabs.forEach(t=>{const on=t.dataset.t===name;t.classList.toggle('on',on);t.setAttribute('aria-selected',on?'true':'false');t.tabIndex=on?0:-1;});
  panels.forEach(p=>p.classList.toggle('on',p.dataset.p===name));
  store.set('dash-tab',name);
  if(!keepScroll)toTop();
}
window.dashGo=go;
tabs.forEach((t,i)=>{t.addEventListener('click',()=>go(t.dataset.t));
  t.addEventListener('keydown',ev=>{if(ev.key==='ArrowRight'||ev.key==='ArrowLeft'){const n=tabs[(i+(ev.key==='ArrowRight'?1:tabs.length-1))%tabs.length];go(n.dataset.t);n.focus();}});});
go((location.hash||'').replace('#','')||store.get('dash-tab')||'overview',true);

/* theme and snapshot age */
const th=store.get('dash-theme');if(th==='dark'||th==='light')root.dataset.theme=th;
$('#dash-theme').addEventListener('click',()=>{const n=root.dataset.theme==='dark'?'light':'dark';root.dataset.theme=n;store.set('dash-theme',n);});
const ag=$('#dash-age');
function age(){const m=Math.max(0,Math.round((Date.now()/1000-(+ag.dataset.gen))/60));
  ag.textContent=' · '+(m<1?'just now':m<60?m+' min ago':m<1440?Math.floor(m/60)+' h '+(m%60)+' min ago':Math.floor(m/1440)+' d ago');
  const stale=m>=90;ag.parentElement.classList.toggle('stale',stale);ag.parentElement.title=stale?'This snapshot is more than 90 minutes old. Run refreshpage to rebuild it.':'Static snapshot. Run refreshpage to rebuild it.';}
age();setInterval(age,60000);

/* results filters */
let fm='';const fc=$('#f-cat'),fq=$('#f-q'),fch=$('#f-chart'),cards=$$('.entry');
function apply(){const c=fc.value,q=fq.value.trim().toLowerCase(),ch=fch.checked;let n=0;
  cards.forEach(el=>{const ok=(!fm||el.dataset.models.split(' ').includes(fm))&&(!c||el.dataset.cat===c)&&(!ch||el.dataset.charts!=='0')&&(!q||el.textContent.toLowerCase().includes(q));
    el.hidden=!ok;if(ok)n++;});
  $$('.day').forEach(d=>{const k=$$('.entry',d).filter(e=>!e.hidden).length;d.hidden=!k;$('.dayh .n',d).textContent=k+(k===1?' run':' runs');});
  $('#f-count').textContent=n+' of '+cards.length+' runs';$('#f-none').hidden=n>0;
  $$('.pill[data-m]',$('#f-models')).forEach(p=>p.classList.toggle('on',p.dataset.m===fm));
  $$('.matrix td.cell').forEach(td=>td.classList.toggle('sel',!!fm&&!!c&&td.dataset.m===fm&&td.dataset.c===c));}
function setFilters(m,c,q,ch){fm=m;fc.value=c;fq.value=q;fch.checked=ch;apply();}
$$('.pill[data-m]',$('#f-models')).forEach(p=>p.addEventListener('click',()=>{fm=p.dataset.m;apply();}));
[fc,fch].forEach(e=>e.addEventListener('change',apply));fq.addEventListener('input',apply);
$('#f-clear').addEventListener('click',()=>setFilters('','','',false));
function openEntry(id){go('results',true);setFilters('','','',false);const el=document.getElementById(id);if(!el)return;
  const d=el.querySelector('details');if(d)d.open=true;if(!managed)el.scrollIntoView({block:'start'});el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1800);}

/* chart gallery filters */
let gm='',gg='';const gt=$('#g-tied'),tiles=$$('.tile');
function gapply(){let n=0;tiles.forEach(t=>{const ok=(!gm||t.dataset.models.split(' ').includes(gm))&&(!gg||t.dataset.g===gg)&&(!gt.checked||t.dataset.tied==='1');t.hidden=!ok;if(ok)n++;});
  $$('.gsec').forEach(s=>{const k=$$('.tile',s).filter(t=>!t.hidden).length;s.hidden=!k;$('.gsech .n',s).textContent=k+(k===1?' chart':' charts');});
  $('#g-count').textContent=n+' of '+tiles.length+' charts';
  $$('.pill[data-m]',$('#g-models')).forEach(p=>p.classList.toggle('on',p.dataset.m===gm));
  $$('.pill[data-g]',$('#g-groups')).forEach(p=>p.classList.toggle('on',p.dataset.g===gg));}
$$('.pill[data-m]',$('#g-models')).forEach(p=>p.addEventListener('click',()=>{gm=p.dataset.m;gapply();}));
$$('.pill[data-g]',$('#g-groups')).forEach(p=>p.addEventListener('click',()=>{gg=p.dataset.g;gapply();}));
gt.addEventListener('change',gapply);

/* lightbox */
const lb=$('#dash-lb'),lbImg=$('#dash-lb-img'),stage=$('#dash-lb-stage'),lbPath=$('#dash-lb-path'),lbEntry=$('#dash-lb-entry'),lbCopy=$('#dash-lb-copy');
let lbList=[],lbI=0;
function show(i){lbI=(i+lbList.length)%lbList.length;const im=lbList[lbI];lbImg.src=IMG[im.dataset.img];stage.classList.remove('zoom');stage.scrollTop=0;
  $('#dash-lb-title').textContent=im.dataset.title||'';$('#dash-lb-n').textContent=lbList.length>1?(lbI+1)+' / '+lbList.length:'';
  lbPath.textContent='code '+im.dataset.path;lbEntry.hidden=!im.dataset.entry;lbEntry.dataset.e=im.dataset.entry||'';
  $$('.lb-nav').forEach(b=>b.hidden=lbList.length<2);}
function openLb(list,i,anchor){lbList=list;
  if(managed){const r=root.getBoundingClientRect(),a=(anchor||list[i]).getBoundingClientRect();
    lb.style.top=Math.max(0,Math.min(a.top-r.top-90,root.offsetHeight-770))+'px';}
  lb.classList.add('on');show(i);}
function closeLb(){lb.classList.remove('on');lbImg.removeAttribute('src');}
lbImg.addEventListener('click',ev=>{ev.stopPropagation();stage.classList.toggle('zoom');});
stage.addEventListener('click',closeLb);
$('#dash-lb-close').addEventListener('click',closeLb);
$('#dash-lb-prev').addEventListener('click',()=>show(lbI-1));
$('#dash-lb-next').addEventListener('click',()=>show(lbI+1));
lbEntry.addEventListener('click',()=>{const id=lbEntry.dataset.e;closeLb();openEntry(id);});
lbCopy.addEventListener('click',()=>{const txt=lbPath.textContent;
  const sel=()=>{const r=document.createRange();r.selectNodeContents(lbPath);const s=getSelection();s.removeAllRanges();s.addRange(r);};
  const done=()=>{lbCopy.textContent='Copied';setTimeout(()=>{lbCopy.textContent='Copy command';},1300);};
  if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(txt).then(done,sel);else sel();});

/* one click handler for everything that navigates */
root.addEventListener('click',ev=>{const t=ev.target;if(!t.closest||lb.contains(t))return;
  const th=t.closest('.thumb');
  if(th){const list=$$('.figs img[data-img]',th.closest('.entry'));if(list.length)openLb(list,0,th);return;}
  const ph=t.closest('.tile .ph');const im=ph?$('img[data-img]',ph):(t.matches('img[data-img]')?t:null);
  if(im){const box=im.closest('.figs,.gallery');let list=box?$$('img[data-img]',box).filter(x=>x.offsetParent!==null):[im];let i=list.indexOf(im);if(i<0){list=[im];i=0;}openLb(list,i,im);return;}
  const j=t.closest('[data-e].jump');if(j){ev.preventDefault();openEntry(j.dataset.e);return;}
  const g=t.closest('[data-go]');if(g){go(g.dataset.go);return;}
  const mc=t.closest('.matrix [data-m],.matrix [data-c]');
  if(mc){go('results',true);setFilters(mc.dataset.m||'',mc.dataset.c||'','',false);toTop();}});

document.addEventListener('keydown',ev=>{
  if(lb.classList.contains('on')){if(ev.key==='Escape')closeLb();else if(ev.key==='ArrowLeft'&&lbList.length>1)show(lbI-1);else if(ev.key==='ArrowRight'&&lbList.length>1)show(lbI+1);return;}
  const tg=ev.target&&ev.target.tagName||'';if(/INPUT|SELECT|TEXTAREA/.test(tg)||ev.metaKey||ev.ctrlKey||ev.altKey)return;
  if(ev.key.length===1&&ev.key>='1'&&ev.key<=String(tabs.length)){go(tabs[+ev.key-1].dataset.t);}
  else if(ev.key==='/'){ev.preventDefault();go('results');fq.focus();}});
apply();gapply();
})();
"""

LOGO = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' aria-hidden='true'>"
        "<path d='M5 4v15h15'/><path d='M5 19 16.5 8.5'/><circle cx='17.5' cy='7.5' r='1.6' fill='currentColor' stroke='none'/></svg>")


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def plain(s: str) -> str:
    return re.sub(r"[`*]", "", s)


def badge(state: str) -> str:
    cls, glyph, label = STATE_STYLE.get(state, ("b-plain", "◆", state))
    return f"<span class='badge {cls}'><span class='g'>{glyph}</span>{esc(label)}</span>"


def model_tag(m: str, short: bool = True) -> str:
    cls = m if m in MODEL_LABEL else "methods"
    name = (MODEL_SHORT if short else MODEL_LABEL).get(m, m)
    return f"<span class='mtag'><i class='mdot m-{cls}'></i>{esc(name)}</span>"


def progress_html(pr) -> str:
    if not pr:
        return ""
    done, total, unit = pr
    if not total:
        return f"<div class='prog'><div class='pl'><span><b>{done:,}</b> {esc(unit)}</span></div></div>"
    pct = min(100, round(100 * done / total))
    right = "✓ done" if done >= total else f"{pct}%"
    return (f"<div class='prog'><div class='pl'><span><b>{done:,}</b> of {total:,} {esc(unit)}</span><span class='pp'>{right}</span></div>"
            f"<div class='bar' role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='{pct}'><i style='width:{pct}%'></i></div></div>")


def short_date(d: str) -> str:
    try:
        return dt.datetime.strptime(d[:10], "%Y-%m-%d").strftime("%b %-d")
    except ValueError:
        return d


def long_date(d: str) -> str:
    try:
        return dt.datetime.strptime(d[:10], "%Y-%m-%d").strftime("%A, %B %-d")
    except ValueError:
        return d or "Undated"


def fmt_dur(sec: int) -> str:
    h, m = divmod(sec // 60, 60)
    if h >= 24:
        return f"{h // 24} d {h % 24} h"
    return f"{h} h {m:02d} min" if h else f"{m} min"


def limit_seconds(s: str) -> int:
    try:
        return elapsed_seconds(s)
    except ValueError:
        return 0


def tint(n: int) -> str:
    return "t1" if n <= 1 else "t2" if n == 2 else "t3" if n <= 4 else "t4" if n <= 7 else "t5"


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _write_whole(path, text):
    """Write beside the target and rename over it, so a reader (statuspage copying the notebook, a browser
    reloading the page) never sees a half-written 10 MB file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--findings", default=str(EXP / "FINDINGS.md"))
    ap.add_argument("--out", default=str(EXP / "dashboard" / "status.html"))
    ap.add_argument("--in-progress", default=str(EXP / "dashboard" / "in_progress.json"))
    ap.add_argument("--rescan-charts", action="store_true")
    ap.add_argument("--chart", action="append", default=[], help="extra chart PNG to index (repeatable)")
    ap.add_argument("--no-squeue", action="store_true")
    args = ap.parse_args()

    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    status_md, entries = parse_findings(Path(args.findings))
    prog_cfg = {"items": [], "note": ""}
    if os.path.exists(args.in_progress):
        prog_cfg = json.loads(Path(args.in_progress).read_text())
    items = prog_cfg.get("items", [])
    extra = list(args.chart) + [os.path.expandvars(c) for it in items for c in it.get("charts", [])]
    charts = chart_index(out.parent / "charts_index.txt", args.rescan_charts, extra)
    queue, queue_err = read_queue(args.no_squeue)
    qmap = {g["id"]: g for g in (queue or [])}

    # ---- images: stored once in a JSON block, attached by the page script when they scroll into view
    images: dict[str, str] = {}
    img_id: dict[str, str] = {}
    dims: dict[str, tuple[int, int]] = {}

    def use_img(path: str) -> str:
        if path not in img_id:
            try:
                uri, w, h = embed_png(path)
            except Exception as exc:  # noqa: BLE001
                print(f"cannot embed {path}: {exc}", file=sys.stderr)
                return ""
            img_id[path] = f"i{len(img_id)}"
            images[img_id[path]] = uri
            dims[path] = (w, h)
        return img_id[path]

    def img_tag(path: str, entry: dict | None) -> str:
        iid = use_img(path)
        if not iid:
            return ""
        title = plain(entry["title"]) if entry else os.path.basename(path)
        w, h = dims[path]
        return (f"<img data-img='{iid}' data-title='{esc(title)}' data-path='{esc(path)}' data-entry='{entry['id'] if entry else ''}' "
                f"width='{w}' height='{h}' alt='{esc(title)}'>")

    used: set[str] = set()
    for e in entries:
        e["charts"], e["missing"] = [], []
        for ref in dict.fromkeys(PNG_RE.findall(e["headline"] + "\n" + e["body_md"])):
            p = resolve_chart(ref, charts, e["models"])
            if p and p not in e["charts"]:
                e["charts"].append(p)
                used.add(p)
            elif not p:
                e["missing"].append(ref)
    by_date = sorted(entries, key=lambda e: (e["date"][:10], -e["order"]), reverse=True)
    chart_entry: dict[str, dict] = {}
    for e in by_date:
        for p in e["charts"]:
            chart_entry.setdefault(p, e)

    now = time.strftime("%-I:%M %p %Z")
    today = dt.date.today()
    now_full = time.strftime("%Y-%m-%d %-I:%M %p %Z")
    try:
        head = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h", "--date=format:%b %-d %-I:%M %p"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        head_when = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cd", "--date=format:%b %-d, %-I:%M %p"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
        branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        head, head_when, branch = "", "", ""

    # ---- in-progress items: state, progress, queue lines
    for it in items:
        it["_state"] = item_state(it, queue)
        it["_prog"] = progress_of(it.get("progress"))
    order = {s: i for i, s in enumerate(ACTIVE_STATES)}
    items = sorted(items, key=lambda it: order.get(it["_state"], len(order)))  # stable: file order within a state
    active = [it for it in items if it["_state"] in order]
    parked = [it for it in items if it["_state"] not in order]

    def job_lines(it: dict) -> str:
        bits = []
        for j in it.get("jobs", []):
            g = qmap.get(str(j))
            if queue is None:
                bits.append(f"<span class='mono'>{esc(j)}</span>")
            elif g:
                parts = [f"{g['running']} running" if g["running"] else "", f"{g['pending']} pending" if g["pending"] else ""]
                parts += [f"{v} {k.lower()}" for k, v in g["other"].items()]
                bits.append(f"<span class='mono'>{esc(j)}</span> · {esc(', '.join(p for p in parts if p))}")
            else:
                bits.append(f"<span class='mono'>{esc(j)}</span> · no longer in the queue")
        return "<br>".join(bits)

    def item_card(it: dict) -> str:
        models = "".join(f"<span class='chip'>{model_tag(m, short=False)}</span>" for m in it.get("models", []))
        cat = it.get("category")
        cat_chip = f"<span class='chip cat'>{esc(CAT_LABEL.get(cat, cat))}</span>" if cat else ""
        kv = []
        if it.get("eta"):
            kv.append(f"<dt>{'Expected' if it['_state'] in ACTIVE_STATES else 'Next'}</dt><dd>{esc(it['eta'])}</dd>")
        jl = job_lines(it)
        if jl:
            kv.append(f"<dt>Jobs</dt><dd>{jl}</dd>")
        mk = last_line(it.get("marker_file"))
        if mk:
            kv.append(f"<dt>Marker</dt><dd class='mono'>{esc(mk)}</dd>")
        return (f"<div class='card wcard'><div><div class='hd'><h3>{esc(it.get('label', ''))}</h3>{badge(it['_state'])}</div>"
                f"<div class='chips' style='margin-top:7px'>{models}{cat_chip}</div>"
                f"<p class='det'>{esc(it.get('detail', ''))}</p></div>"
                f"<div class='side'>{progress_html(it['_prog'])}<dl class='kv'>{''.join(kv)}</dl></div></div>")

    P: list[str] = []
    P.append("<section class='panel' data-p='progress'><h2 class='ph'>In progress</h2>")
    if prog_cfg.get("note"):
        P.append(f"<div class='note'>{md.render(prog_cfg['note'])}</div>")
    if not items:
        P.append("<p class='empty-note'>Nothing listed in in_progress.json.</p>")
    if active:
        P.append(f"<h2 class='sec'>Running and queued <span class='n'>{plural(len(active), 'item')}</span></h2>" + "".join(item_card(it) for it in active))
    if parked:
        P.append(f"<h2 class='sec'>Awaiting write-up <span class='n'>{plural(len(parked), 'item')}, not yet in FINDINGS.md</span></h2>"
                 + "".join(item_card(it) for it in parked))
    run_tasks = sum(g["running"] for g in queue or [])
    pend_tasks = sum(g["pending"] for g in queue or [])
    P.append("<h2 class='sec'>Slurm queue <span class='n'>"
             + (esc(queue_err) if queue is None else f"squeue at {esc(now)} · {plural(run_tasks, 'task')} running, {pend_tasks:,} pending") + "</span></h2>")
    if queue:
        labels = {str(j): it.get("label", "") for it in items for j in it.get("jobs", [])}
        P.append("<div class='card tblwrap'><table class='tbl'><tr><th>Job</th><th>For</th><th>Partition / qos</th><th>Tasks</th>"
                 "<th>Longest elapsed of limit</th><th>Waiting on</th></tr>")
        for g in sorted(queue, key=lambda g: g["id"]):
            n_other = sum(g["other"].values())
            seg = "".join(f"<i class='{c}' style='flex:{n}'></i>" for c, n in (("r", g["running"]), ("p", g["pending"])) if n)
            counts = [f"● {g['running']} running" if g["running"] else "", f"○ {g['pending']} pending" if g["pending"] else ""]
            counts += [f"{v} {k.lower()}" for k, v in g["other"].items()]
            tasks = (f"<div class='seg' role='img' aria-label='{g['running']} running, {g['pending']} pending'>{seg}</div>" if seg else "") \
                + f"<div class='small'>{esc(' · '.join(c for c in counts if c))}</div>"
            if g["nodes"]:
                tasks += f"<div class='small mono' style='margin-top:2px'>{esc(', '.join(sorted(g['nodes'])))}</div>"
            el, lim = elapsed_seconds(g["elapsed"]), limit_seconds(g["limit"])
            if el and lim:
                timing = (f"{esc(fmt_dur(el))} <span class='small'>of {esc(fmt_dur(lim))}</span>"
                          f"<div class='meter'><i style='width:{min(100, round(100 * el / lim))}%'></i></div>")
            elif lim:
                timing = f"<span class='small'>not started · limit {esc(fmt_dur(lim))}</span>"
            else:
                timing = f"<span class='small'>{esc(g['elapsed'] or '–')} / {esc(g['limit'])}</span>"
            wait = esc(", ".join(sorted(g["reasons"])))
            if g["start"] and fmt_start(g["start"]):
                wait += f"<div class='small'>scheduler estimate {esc(fmt_start(g['start']))} (worst case)</div>"
            P.append(f"<tr><td><span class='mono'>{esc(g['id'])}</span><div class='jn'>{esc(g['name'])}</div></td><td>{esc(labels.get(g['id'], ''))}</td>"
                     f"<td>{esc(', '.join(sorted(g['parts'])))}<div class='small'>qos {esc(g['qos'])}</div></td><td>{tasks}</td><td>{timing}</td><td>{wait}</td></tr>")
            del n_other
        P.append("</table></div>")
    elif queue is not None:
        P.append(f"<p class='small'>No jobs in the queue at {esc(now)}.</p>")
    P.append("</section>")
    panel_progress = "".join(P)

    # ---- matrix (overview)
    cell: dict[tuple[str, str], list[dict]] = {}
    per_model: dict[str, int] = {}
    for e in entries:
        for m in e["models"]:
            cell.setdefault((e["cat"], m), []).append(e)
            per_model[m] = per_model.get(m, 0) + 1
    wip: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        for m in it.get("models", []):
            wip.setdefault((it.get("category", "other"), m), []).append(it)
    M = ["<div class='mwrap'><table class='matrix'><thead><tr><th class='rh' style='cursor:default'></th>"]
    for m, _lab in MODELS:
        M.append(f"<th class='mh' data-m='{m}' title='List every run for this model'>{model_tag(m)}<span class='n'>{plural(per_model.get(m, 0), 'run')}</span></th>")
    M.append("</tr></thead><tbody>")
    for c in CAT_ORDER:
        if not any((c, m) in cell or (c, m) in wip for m, _ in MODELS):
            continue
        row = [f"<tr><th class='rh' data-c='{c}' title='List every run of this kind'>{esc(CAT_LABEL[c])}</th>"]
        for m, _lab in MODELS:
            es, ws = cell.get((c, m), []), wip.get((c, m), [])
            if not es and not ws:
                row.append("<td class='empty'>·</td>")
                continue
            bits = ""
            if es:
                latest = max(e["date"][:10] for e in es)
                bits += f"<span class='n'>{len(es)}</span><span class='d'><span class='lw'>latest </span>{esc(short_date(latest))}</span>"
            if ws:
                bits += "<div class='w'>" + "".join(badge(it["_state"]) for it in ws) + "</div>"
            tip = f"{MODEL_LABEL[m]}, {CAT_LABEL[c]}: {plural(len(es), 'recorded run')}" + (f", {len(ws)} in progress" if ws else "")
            row.append(f"<td class='cell {tint(len(es)) if es else ''}' data-m='{m}' data-c='{c}' title='{esc(tip)}'>{bits}</td>")
        M.append("".join(row) + "</tr>")
    M.append("</tbody></table></div>")
    ramp = "".join(f"<i style='background:color-mix(in srgb,var(--accent) var(--t{i}),var(--card))'></i>" for i in range(1, 6))
    M.append(f"<div class='legend'><span>Number = analysed runs recorded in FINDINGS.md. Shade: 1 <span class='ramp'>{ramp}</span> 8 or more</span>"
             f"<span>{badge('running')} {badge('queued')} {badge('interim result')} work in progress</span>"
             "<span>Click a cell, a model or an experiment to list its runs.</span></div>")
    matrix_html = "".join(M)

    # ---- results panel
    R = ["<section class='panel' data-p='results'><h2 class='ph'>Results</h2><div class='filters'><div class='pills' id='f-models'>"
         f"<button class='pill on' data-m=''>All models <span class='c'>{len(entries)}</span></button>"]
    for m, _lab in MODELS:
        R.append(f"<button class='pill' data-m='{m}'><i class='mdot m-{m}'></i>{esc(MODEL_SHORT[m])} <span class='c'>{per_model.get(m, 0)}</span></button>")
    cat_n: dict[str, int] = {}
    for e in entries:
        cat_n[e["cat"]] = cat_n.get(e["cat"], 0) + 1
    R.append("</div><div class='frow'><select id='f-cat' aria-label='Experiment kind'><option value=''>All experiments</option>"
             + "".join(f"<option value='{c}'>{esc(CAT_LABEL[c])} ({cat_n[c]})</option>" for c in CAT_ORDER if cat_n.get(c)) + "</select>"
             "<input type='search' id='f-q' placeholder='Search titles and full text' aria-label='Search runs'>"
             "<label><input type='checkbox' id='f-chart'> with charts</label>"
             "<button class='btn' id='f-clear'>Clear</button><span class='count' id='f-count'></span></div></div>")
    day = None
    for e in by_date:
        d = e["date"][:10]
        if d != day:
            if day is not None:
                R.append("</section>")
            R.append(f"<section class='day'><h3 class='dayh'>{esc(long_date(d))} <span class='n'></span></h3>")
            day = d
        chips = "".join(f"<span class='chip'>{model_tag(m)}</span>" for m in e["models"])
        chips += f"<span class='chip cat'>{esc(CAT_LABEL[e['cat']])}</span>"
        if " to " in e["date"]:
            chips += f"<span class='chip'>{esc(e['date'])}</span>"
        if e["charts"]:
            chips += f"<span class='chip'>{plural(len(e['charts']), 'chart')}</span>"
        figs = "".join(f"<figure>{img_tag(p, e)}<figcaption><code>code {esc(p)}</code></figcaption></figure>" for p in e["charts"] if use_img(p))
        thumb = ""
        if e["charts"] and use_img(e["charts"][0]):
            w, h = dims[e["charts"][0]]
            thumb = (f"<button class='thumb' title='Open the chart'><img data-img='{img_id[e['charts'][0]]}' width='{w}' height='{h}' "
                     f"alt='Chart preview: {esc(plain(e['title']))}'></button>")
        miss = (f"<div class='small'>Charts named but not found in the index: {esc(', '.join(e['missing']))}</div>" if e["missing"] else "")
        full_md = e["headline"] + ("\n\n" + e["body_md"] if e["body_md"] else "")
        R.append(f"<article class='card entry' id='{e['id']}' data-models='{' '.join(e['models'])}' data-cat='{e['cat']}' data-charts='{len(e['charts'])}'>"
                 f"<div><h3>{md.renderInline(e['title'])}</h3><div class='chips'>{chips}</div><p class='sum'>{md.renderInline(e['summary'])}</p></div>{thumb}"
                 f"<details><summary>Full entry{' and charts' if e['charts'] else ''}</summary><div class='body'>"
                 f"{('<div class=figs>' + figs + '</div>') if figs else ''}{miss}{md.render(full_md)}</div></details></article>")
    if day is not None:
        R.append("</section>")
    R.append("<p class='empty-note' id='f-none' hidden>No recorded run matches these filters.</p></section>")
    panel_results = "".join(R)

    # ---- charts panel: charts tied to a result first (newest result first), then the rest by file time
    tied = [p for e in by_date for p in e["charts"]]
    tied = list(dict.fromkeys(tied))
    rest = sorted((p for p in charts if p not in used), key=os.path.getmtime, reverse=True)
    group_of = chart_groups(tied + rest, chart_entry)
    T: dict[str, list[tuple[str, str]]] = {k: [] for k, _, _ in CHART_GROUPS}
    g_models: dict[str, int] = {}
    for p in tied + rest:
        e = chart_entry.get(p)
        tag = img_tag(p, e)
        if not tag:
            continue
        models = e["models"] if e else [chart_model(p)]
        for m in models:
            g_models[m] = g_models.get(m, 0) + 1
        if e:
            title = md.renderInline(e["title"])
            sub = (f"<span>{esc(short_date(e['date']))}</span>" + "".join(model_tag(m) for m in models)
                   + f"<button class='link jump' data-e='{e['id']}'>Open result</button>")
        else:
            title = esc(os.path.basename(p))
            folder = "/".join(Path(p).parent.parts[-2:])
            when = dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%b %-d")
            sub = f"<span>{esc(when)}</span>" + "".join(model_tag(m) for m in models) + f"<span class='mono' title='{esc(p)}'>{esc(folder)}</span>"
        g = group_of[p]
        if g == "main":  # say which of the three kinds a tile is: result titles and file names do not always
            sub = f"<span class='chip'>{MAIN_KIND[main_group_key(p)[0]]}</span>" + sub
        T[g].append((p, f"<figure class='tile' data-models='{' '.join(models)}' data-tied='{1 if e else 0}' data-g='{g}'><div class='ph'>{tag}</div>"
                        f"<figcaption><div class='tt'>{title}</div><div class='ts'>{sub}</div></figcaption></figure>"))
    T["main"].sort(key=lambda t: main_group_key(t[0]))
    n_tiles = sum(len(v) for v in T.values())
    G = ["<section class='panel' data-p='charts'><h2 class='ph'>Charts</h2><div class='gbar'><div class='pills' id='g-models'>"
         f"<button class='pill on' data-m=''>All models <span class='c'>{n_tiles}</span></button>"]
    for m, _lab in MODELS:
        if g_models.get(m):
            G.append(f"<button class='pill' data-m='{m}'><i class='mdot m-{m}'></i>{esc(MODEL_SHORT[m])} <span class='c'>{g_models[m]}</span></button>")
    G.append("</div><label><input type='checkbox' id='g-tied'> only charts named in a result</label><span class='count' id='g-count'></span></div>"
             f"<div class='gbar gbar2'><div class='pills' id='g-groups'><button class='pill on' data-g=''>All groups <span class='c'>{n_tiles}</span></button>")
    for k, lab, _desc in CHART_GROUPS:
        if T[k]:
            G.append(f"<button class='pill' data-g='{k}'>{esc(lab.split(':')[0])} <span class='c'>{len(T[k])}</span></button>")
    G.append("</div></div>")
    for k, lab, desc in CHART_GROUPS:
        if T[k]:
            G.append(f"<section class='gsec' data-g='{k}'><h3 class='gsech'>{esc(lab)} <span class='n'>{plural(len(T[k]), 'chart')}</span></h3>"
                     f"<p class='gdesc'>{esc(desc)}</p><div class='gallery'>" + "".join(t for _, t in T[k]) + "</div></section>")
    G.append("</section>")
    panel_charts = "".join(G)

    # ---- status log panel: each dated update of "Current status" becomes one collapsible line
    L = ["<section class='panel' data-p='log'><h2 class='ph'>Status log</h2>"]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", status_md) if b.strip()]
    dated, other = [], []
    for b in blocks:
        m = re.match(r"\*\*(.+?)\*\*\s*(.*)$", b, re.S)
        dm = re.match(r"(.+?)[\s(]+(20\d\d-\d\d-\d\d)(.*)$", m.group(1), re.S) if m else None
        if m and dm and len(dm.group(1)) > 12:
            dated.append((dm.group(2), dm.group(1).strip(" ,;("), b))
        else:
            other.append(b)
    if dated:
        L.append(f"<h2 class='sec'>Dated updates <span class='n'>{len(dated)} from the “Current status” section of FINDINGS.md, newest first</span></h2><div class='card loglist'>")
        for d, lead, body in sorted(dated, key=lambda x: x[0], reverse=True):
            L.append(f"<details class='logi'><summary><span class='d'>{esc(short_date(d))}</span><span>{md.renderInline(lead)}</span></summary>"
                     f"<div class='body'>{md.render(body)}</div></details>")
        L.append("</div>")
    if other:
        L.append("<h2 class='sec'>Standing summary <span class='n'>the study-level status text and question table</span></h2>"
                 f"<div class='card logrest'><div class='body' style='border-top:0;margin-top:0;padding-top:0'>{md.render(chr(10).join(x + chr(10) for x in other))}</div></div>")
    L.append("</section>")
    panel_log = "".join(L)

    # ---- overview panel
    week = sum(1 for e in entries if e["date"][:10] >= (today - dt.timedelta(days=6)).isoformat())
    latest_date = max((e["date"][:10] for e in entries), default="")
    if queue is None:
        k_run = ("–", "", esc(queue_err))
        k_pend = ("–", "", "")
    else:
        longest = max((elapsed_seconds(g["elapsed"]) for g in queue), default=0)
        k_run = (f"{run_tasks:,}", "tasks", f"in {plural(sum(1 for g in queue if g['running']), 'job')}" + (f" · longest {fmt_dur(longest)}" if longest else ""))
        starts = sorted(g["start"] for g in queue if g["pending"] and g["start"] and fmt_start(g["start"]))
        k_pend = (f"{pend_tasks:,}", "tasks", (f"scheduler estimate {fmt_start(starts[0])}" if starts else "nothing waiting" if not pend_tasks else "no start estimate yet"))
    nxt = next((it.get("eta_short") for it in active if it.get("eta_short")), "")
    k_wip = (str(len(active)), "active", (f"next {nxt}" if nxt else "") + (f"{' · ' if nxt else ''}{len(parked)} to write up" if parked else ""))
    tiles_kpi = [
        ("progress", "<span class='kg run'>●</span>Running on the cluster", *k_run),
        ("progress", "<span class='kg q'>●</span>Queued", *k_pend),
        ("progress", "Work in progress", *k_wip),
        ("results", "Analysed runs recorded", f"{len(entries):,}", "", f"{week} in the last 7 days · latest {short_date(latest_date)}"),
        ("charts", "Charts", f"{n_tiles:,}", "", f"{len(tied)} named in a result"),
    ]
    O = ["<section class='panel on' data-p='overview'><h2 class='ph'>Overview</h2><div class='kpis'>"]
    for go_to, label, val, unit, sub in tiles_kpi:
        O.append(f"<button class='kpi' data-go='{go_to}'><div class='l'>{label}</div><div class='v'>{val}{f'<small>{esc(unit)}</small>' if unit else ''}</div>"
                 f"<div class='s' title='{esc(sub)}'>{esc(sub) or '&nbsp;'}</div></button>")
    O.append("</div><div class='grid2'><div class='stack'><div class='card'><div class='ch'><h2>In progress</h2>"
             "<button class='link' data-go='progress'>Details and queue →</button></div><div class='cb'>")
    for it in items:
        models = "".join(model_tag(m) for m in it.get("models", []))
        eta = f"<span>{esc(it['eta_short'])}</span>" if it.get("eta_short") else ""
        O.append(f"<div class='wrow'><div><div class='t'>{esc(it.get('label', ''))}</div><div class='u'>{models}{eta}</div></div>"
                 f"<div>{progress_html(it['_prog'])}</div><div class='st'>{badge(it['_state'])}</div></div>")
    if not items:
        O.append("<p class='small'>Nothing listed in in_progress.json.</p>")
    O.append("</div></div><div class='card'><div class='ch'><h2>Slurm queue</h2><span class='hint'>"
             + (esc(queue_err) if queue is None else f"squeue at {esc(now)}") + "</span><button class='link' data-go='progress'>Full table →</button></div><div class='cb'>")
    labels = {str(j): it.get("label", "") for it in items for j in it.get("jobs", [])}
    for g in sorted(queue or [], key=lambda g: g["id"]):
        seg = "".join(f"<i class='{c}' style='flex:{n}'></i>" for c, n in (("r", g["running"]), ("p", g["pending"])) if n)
        counts = " · ".join(x for x in (f"● {g['running']} running" if g["running"] else "", f"○ {g['pending']} pending" if g["pending"] else "") if x)
        el, lim = elapsed_seconds(g["elapsed"]), limit_seconds(g["limit"])
        when = (f"{fmt_dur(el)}<br>of {fmt_dur(lim)}" if el and lim else f"est. start<br>{fmt_start(g['start'])}" if fmt_start(g["start"]) else "waiting")
        O.append(f"<div class='qrow'><div><div class='t' title='{esc(labels.get(g['id'], g['name']))}'>{esc(labels.get(g['id'], '') or g['name'])}</div>"
                 f"<div class='u'><span class='mono'>{esc(g['id'])}</span> · {esc(', '.join(sorted(g['parts'])))}</div></div>"
                 f"<div><div class='seg' role='img' aria-label='{g['running']} running, {g['pending']} pending'>{seg}</div><div class='u'>{esc(counts)}</div></div>"
                 f"<div class='el'>{when}</div></div>")
    if queue is not None and not queue:
        O.append("<p class='small'>No jobs in the queue.</p>")
    O.append("</div></div></div><div class='card'><div class='ch'><h2>Latest results</h2><button class='link' data-go='results'>All "
             f"{len(entries)} runs →</button></div><div class='cb'>")
    for e in by_date[:6]:
        O.append(f"<a class='lrow jump' href='#{e['id']}' data-e='{e['id']}'><span class='d'>{esc(short_date(e['date']))}</span>"
                 f"<span class='t'>{md.renderInline(e['title'])}</span><span class='s'>{md.renderInline(first_sentences(e['headline'], 230))}</span>"
                 f"<span class='m'>{''.join(model_tag(m) for m in e['models'])}<span>{esc(CAT_LABEL[e['cat']])}</span></span></a>")
    O.append("</div></div></div><div class='card'><div class='ch'><h2>Results by model and experiment</h2>"
             "<span class='hint'>what has been run where</span></div>" + matrix_html + "</div></section>")
    panel_overview = "".join(O)

    # ---- page
    tabs = [("overview", "Overview", ""), ("progress", "In progress", str(len(items))), ("results", "Results", str(len(entries))),
            ("charts", "Charts", str(n_tiles)), ("log", "Status log", "")]
    nav = ("<nav class='tabs' role='tablist' aria-label='Sections'><div class='in'>"
           + "".join(f"<button class='tab{' on' if i == 0 else ''}' role='tab' data-t='{k}'>{esc(lab)}{f'<span class=ct>{ct}</span>' if ct else ''}</button>"
                     for i, (k, lab, ct) in enumerate(tabs)) + "</div></nav>")
    header = (f"<header class='top in'><div class='logo'>{LOGO}</div><div><h1>Schemer-axis study</h1>"
              f"<div class='meta'>branch <span class='mono'>{esc(branch)}</span> @ <span class='mono'>{esc(head)}</span> ({esc(head_when)})</div></div>"
              f"<div class='snap'><span class='stamp'><i class='sd'></i>Snapshot <b>{esc(now)}</b> {esc(today.strftime('%b %-d'))}"
              f"<span id='dash-age' data-gen='{int(time.time())}'></span></span>"
              "<button class='iconbtn' id='dash-theme' title='Switch between light and dark' aria-label='Switch between light and dark'>◐</button></div></header>")
    footer = ("<div class='foot'><span>Static snapshot built by <code>scripts/status_dashboard.py</code> from FINDINGS.md, the Slurm queue and "
              "<code>dashboard/in_progress.json</code>. Rebuild with <code>refreshpage</code>.</span>"
              "<span>Keys: <kbd>1</kbd>–<kbd>5</kbd> switch tabs, <kbd>/</kbd> searches results, <kbd>Esc</kbd> closes a chart.</span></div>")
    lightbox = ("<div class='lb' id='dash-lb' role='dialog' aria-label='Chart viewer'><div class='lb-top'><div class='lb-title' id='dash-lb-title'></div>"
                "<span class='lb-n' id='dash-lb-n'></span><button id='dash-lb-entry'>Open result</button>"
                "<button id='dash-lb-close' aria-label='Close'>Close ✕</button></div>"
                "<button class='lb-nav prev' id='dash-lb-prev' aria-label='Previous chart'>‹</button>"
                "<div class='lb-stage' id='dash-lb-stage'><img id='dash-lb-img' alt=''></div>"
                "<button class='lb-nav next' id='dash-lb-next' aria-label='Next chart'>›</button>"
                "<div class='lb-bot'><span>Open the original:</span><code id='dash-lb-path'></code><button id='dash-lb-copy'>Copy command</button></div></div>")
    fragment = (f"<style>{CSS}</style><div class='dash' data-theme='light'>{header}{nav}<main class='in'>"
                + panel_overview + panel_progress + panel_results + panel_charts + panel_log + footer + "</main>" + lightbox
                + "</div><script type='application/json' id='dash-imgdata'>" + json.dumps(images) + f"</script><script>{JS}</script>")
    doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Schemer-axis status · {esc(now_full)}</title></head><body class='dash-page'>" + fragment + "</body></html>")
    _write_whole(out, doc)
    # The same page as the saved output of a one-cell notebook: VS Code shows a notebook's stored HTML output
    # when the file is opened, with no kernel and nothing served.
    nb = {"cells": [
              {"cell_type": "markdown", "metadata": {}, "source": [
                  f"Schemer-axis status, generated {now_full} by `scripts/status_dashboard.py`. The page is the saved output of the cell below; "
                  "there is nothing to run. Rebuild with `refreshpage` (or, from the repo root, "
                  "`OPENSSL_CONF=/dev/null .venv-313/bin/python experiments/deceiver-axis/scripts/status_dashboard.py`)."]},
              {"cell_type": "code", "execution_count": 1, "metadata": {}, "source": ["# saved output only; rebuild with refreshpage"],
               "outputs": [{"output_type": "display_data", "metadata": {},
                            "data": {"text/html": [fragment], "text/plain": ["schemer-axis status page (HTML output)"]}}]}],
          "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    _write_whole(out.with_suffix(".ipynb"), json.dumps(nb))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(entries)} entries, {len(images)} charts embedded "
          f"({len(used)} tied to entries), {len(items)} in-progress items, queue groups {len(queue) if queue is not None else 'n/a'}; categories {cat_n}")


if __name__ == "__main__":
    main()
