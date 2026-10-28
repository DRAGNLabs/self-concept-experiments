"""Recompute a run's summary metrics from its per-query records, without rerunning the model.

Re-derives each record's `top_slot` with the current rule (capital or country) so older runs
match the current metric definitions, then rewrites mean_interventions.json.

Usage (from the repo root):
    python experiments/binding_ids/scripts/rescore.py <run dir> <hf tokenizer name>
    e.g. python experiments/binding_ids/scripts/rescore.py \
        experiments/binding_ids/results/olmo2-7b allenai/OLMo-2-1124-7B
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from selfconcept.binding_ids.evaluate import summarize, top_slot
from selfconcept.binding_ids.tasks import load_rows
from selfconcept.common.paths import experiment_dir

run_dir, tokenizer_name = Path(sys.argv[1]), sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
rows = load_rows(experiment_dir("binding_ids") / "data" / "capitals.jsonl")

records = [json.loads(line) for line in open(run_dir / "mean_interventions_queries.jsonl")]
for record in records:
    row = rows[record["row"]]
    record["attributes"] = row["attributes"]
    top_id = tokenizer.encode(record["top_token"], add_special_tokens=False)[0]
    record["top_slot"] = top_slot(tokenizer, row, top_id)

summary_path = run_dir / "mean_interventions.json"
results = json.loads(summary_path.read_text())
results.update(summarize(records))
summary_path.write_text(json.dumps(results, indent=2) + "\n")
print(f"rescored {len(records)} records -> {summary_path}")
