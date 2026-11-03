"""Build a self-contained HTML labeling page (no server, no dependencies) from labeling_page.html.jinja: the rubric on
the left, one item at a time on the right, home-row keys (a s d f ...) pick a label, progress autosaves in the browser,
and "Download labels" writes labels.jsonl.

    python build_labeling_page.py --items items.jsonl --rubric rubric.md --labels 0,1,2,3 --title in_character --out page.html
"""
import argparse
import json
from pathlib import Path
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class Section(TypedDict):
    title: str
    text: str


class Item(TypedDict):
    item_id: str
    sections: list[Section]


TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "labeling_page.html.jinja"


def read_items(items_path: Path) -> list[Item]:
    return [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]


def build_page(items: list[Item], rubric: str, labels: list[str], title: str) -> str:
    environment = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True, undefined=StrictUndefined, keep_trailing_newline=True)
    return environment.get_template(TEMPLATE_NAME).render(items=items, rubric=rubric, labels=labels, title=title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="comma-separated, in shortcut order (a, s, d, f, ...)")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    page = build_page(read_items(args.items), args.rubric.read_text(), args.labels.split(","), args.title)
    args.out.write_text(page)


if __name__ == "__main__":
    main()
