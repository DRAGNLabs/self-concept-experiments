"""Build a self-contained HTML labeling page (no server, no dependencies): the rubric on the left, one item at a time on
the right, number keys pick a label, progress autosaves in the browser, and "Download labels" writes labels.jsonl.

    python build_labeling_page.py --items items.jsonl --rubric rubric.md --labels 0,1,2,3 --title in_character --out page.html
"""
import argparse
import html
import json
import re
from pathlib import Path
from typing import TypedDict


class Section(TypedDict):
    title: str
    text: str


class Item(TypedDict):
    item_id: str
    sections: list[Section]


PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 8px 12px; border-bottom: 1px solid #ccc; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  main { flex: 1; display: flex; min-height: 0; }
  #rubric { width: 38%; overflow-y: auto; padding: 12px; border-right: 1px solid #ccc; background: #fafafa; }
  #item { flex: 1; overflow-y: auto; padding: 12px; }
  pre { white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, monospace; font-size: 13px; margin: 0; }
  h3 { margin: 14px 0 6px; font-size: 14px; color: #444; }
  #controls { padding: 8px 12px; border-top: 1px solid #ccc; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .label { font-size: 16px; padding: 6px 14px; cursor: pointer; }
  .label.chosen { background: #2b6cb0; color: white; }
  #note { flex: 1; min-width: 200px; height: 2.4em; }
</style></head>
<body>
<header>
  <b>__TITLE__</b>
  <label>Your name: <input id="labeler" size="12"></label>
  <span id="progress"></span>
  <button onclick="move(-1)">&larr; Prev</button>
  <button onclick="move(1)">Next &rarr;</button>
  <button onclick="jumpToFirstUnlabeled()">First unlabeled</button>
  <button onclick="downloadLabels()">Download labels</button>
  <label>Import labels <input type="file" accept=".jsonl" onchange="importLabels(this.files[0])"></label>
</header>
<main><div id="rubric"><pre id="rubricText"></pre></div><div id="item"></div></main>
<div id="controls"><span id="labelButtons"></span>
  <textarea id="note" placeholder="Optional note (saved with the label)"></textarea>
  <span style="color:#666">Keys: 1&ndash;__LABEL_COUNT__ pick a label, &larr;/&rarr; move</span></div>
<script>
const TITLE = __TITLE_JSON__;
const LABELS = __LABELS_JSON__;
const ITEMS = __ITEMS_JSON__;
const STORAGE_KEY = "labels:" + TITLE;
let state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{"labeler": "", "index": 0, "byItemId": {}}');

document.getElementById("rubricText").textContent = __RUBRIC_JSON__;
const labelerInput = document.getElementById("labeler");
labelerInput.value = state.labeler;
labelerInput.oninput = () => { state.labeler = labelerInput.value; save(); };
const noteInput = document.getElementById("note");
noteInput.oninput = () => {
  const itemId = ITEMS[state.index].item_id;
  if (state.byItemId[itemId]) { state.byItemId[itemId].note = noteInput.value; save(); }
};
LABELS.forEach((label, position) => {
  const button = document.createElement("button");
  button.className = "label";
  button.textContent = (position + 1) + ": " + label;
  button.onclick = () => choose(label);
  document.getElementById("labelButtons").appendChild(button);
});

function save() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

function render() {
  const item = ITEMS[state.index];
  const itemDiv = document.getElementById("item");
  itemDiv.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Item " + (state.index + 1) + " of " + ITEMS.length;
  itemDiv.appendChild(heading);
  for (const section of item.sections) {
    const title = document.createElement("h3");
    title.textContent = section.title;
    const body = document.createElement("pre");
    body.textContent = section.text;
    itemDiv.append(title, body);
  }
  itemDiv.scrollTop = 0;
  const entry = state.byItemId[item.item_id];
  document.querySelectorAll(".label").forEach((button, position) =>
    button.classList.toggle("chosen", entry !== undefined && entry.label === LABELS[position]));
  noteInput.value = entry ? entry.note : "";
  const labeledCount = ITEMS.filter(candidate => state.byItemId[candidate.item_id]).length;
  document.getElementById("progress").textContent = labeledCount + "/" + ITEMS.length + " labeled";
}

function choose(label) {
  state.byItemId[ITEMS[state.index].item_id] = { label: label, note: noteInput.value };
  save();
  move(1);
}

function move(step) {
  state.index = Math.min(ITEMS.length - 1, Math.max(0, state.index + step));
  save();
  render();
}

function jumpToFirstUnlabeled() {
  const index = ITEMS.findIndex(item => !state.byItemId[item.item_id]);
  if (index === -1) { alert("Everything is labeled. Use Download labels."); return; }
  state.index = index;
  save();
  render();
}

function downloadLabels() {
  const lines = ITEMS.filter(item => state.byItemId[item.item_id]).map(item => JSON.stringify({
    item_id: item.item_id, label: state.byItemId[item.item_id].label,
    note: state.byItemId[item.item_id].note, labeler: state.labeler,
  }));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], { type: "application/jsonl" }));
  link.download = "labels_" + TITLE + "_" + (state.labeler || "anonymous") + ".jsonl";
  link.click();
}

async function importLabels(file) {
  const text = await file.text();
  for (const line of text.split("\\n").filter(line => line.trim())) {
    const record = JSON.parse(line);
    state.byItemId[record.item_id] = { label: record.label, note: record.note || "" };
    if (record.labeler && !state.labeler) { state.labeler = record.labeler; labelerInput.value = record.labeler; }
  }
  save();
  render();
}

document.addEventListener("keydown", event => {
  if (event.target === noteInput || event.target === labelerInput) return;
  const position = parseInt(event.key, 10) - 1;
  if (position >= 0 && position < LABELS.length) choose(LABELS[position]);
  else if (event.key === "ArrowRight") move(1);
  else if (event.key === "ArrowLeft") move(-1);
});

render();
</script></body></html>
"""


def script_safe_json(value: object) -> str:
    return json.dumps(value).replace("</", "<\\/")


def read_items(items_path: Path) -> list[Item]:
    return [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]


def build_page(items: list[Item], rubric: str, labels: list[str], title: str) -> str:
    replacement_by_placeholder = {
        "__TITLE_JSON__": script_safe_json(title),
        "__LABELS_JSON__": script_safe_json(labels),
        "__ITEMS_JSON__": script_safe_json(items),
        "__RUBRIC_JSON__": script_safe_json(rubric),
        "__LABEL_COUNT__": str(len(labels)),
        "__TITLE__": html.escape(title),
    }
    placeholder_pattern = "|".join(re.escape(placeholder) for placeholder in replacement_by_placeholder)
    return re.sub(placeholder_pattern, lambda match: replacement_by_placeholder[match.group(0)], PAGE_TEMPLATE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="comma-separated, in key order (key 1 = first)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    page = build_page(read_items(args.items), args.rubric.read_text(), args.labels.split(","), args.title)
    args.out.write_text(page)


if __name__ == "__main__":
    main()
