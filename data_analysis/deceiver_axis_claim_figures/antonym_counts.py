"""Per-concept proxy hits for the 100 antonym controls on each model, plus gi_v2 target and unsteered, via origin/main's own counting code."""
import json, sys
from pathlib import Path
import subprocess, tempfile

# The controls code lives on origin/main (local main may be behind); extract it to a temp dir.
REF = "origin/main"
_om = Path(tempfile.mkdtemp(prefix="om_"))
_repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
_arc = subprocess.run(["git", "-C", _repo, "archive", REF, "experiments/deceiver-axis/scripts/controls",
                       "experiments/deceiver-axis/data/controls", "src"], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(_om)], input=_arc, check=True)
sys.path[:0] = [str(_om / "experiments/deceiver-axis/scripts/controls"), str(_om / "src")]
import cross_model_percentile as cmp
models = json.loads((_om / "experiments/deceiver-axis/data/controls/cross_model_models.json").read_text())
out = {}
counts = {}
for m in models:
    c = cmp.load_model_counts(m, "proxy"); counts[m["name"]] = c
for m in models:
    concepts = cmp.shared_concepts(counts)
    lifts = cmp.model_lifts(counts[m["name"]], concepts)
    out[m["name"]] = {"counts": counts[m["name"]], "concepts": concepts,
                     "lift_by_concept": lifts["lift_by_concept"].tolist(), "target_lift": float(lifts["target_lift"])}
    c = counts[m["name"]]
    print(m["name"], "episodes/concept", c["episodes_per_concept"], "unsteered", c["unsteered"], "target", c["target"], "target_lift", round(out[m["name"]]["target_lift"], 3))
Path(sys.argv[1]).write_text(json.dumps(out))
