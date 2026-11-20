"""Building blind label sets (items under opaque ids, batch files for LLM labelers) and checking the returned labels."""

import ast
import json
import hashlib
import io
import re
import tokenize
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from selfconcept.codebench.harness import CheckAttempt

CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)(```|$)", re.S)
TRIPLE_QUOTED = re.compile(r"(\"\"\"|''')(.*?)\1", re.S)


class BlindItem(TypedDict):
    item: str
    text: str


class LabelRow(TypedDict):
    """What every labeler output row has; each label set's schema (label_sets.py) adds its labels."""
    item: str


class LabelTally(TypedDict):
    n_items: int
    n_labeled: int
    unlabeled_items: list[str]
    unknown_items: list[str]
    quote_problems: Sequence[Mapping[str, object]]
    counts_by_group: dict[str, dict[str, dict[str, int]]]


def opaque_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:10]


def docstrings(tree: ast.AST) -> list[str]:
    nodes = [tree, *(node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))]
    return [text for node in nodes if (text := ast.get_docstring(node))]  # type: ignore[arg-type]


def code_commentary(code: str) -> str:
    """Comments and docstrings of a code block, in order of appearance (regex fallback when it does not parse)."""
    try:
        tree = ast.parse(code)
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
                    if token.type == tokenize.COMMENT]
        return "\n".join([f'"""{text}"""' for text in docstrings(tree)] + comments)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        lines = [line.strip() for line in code.splitlines() if line.strip().startswith("#")]
        return "\n".join([f'"""{body.strip()}"""' for _, body in TRIPLE_QUOTED.findall(code)] + lines)


def prose_only(reply: str) -> str:
    """The reply with each code block replaced by its comments and docstrings."""
    def replace(match: re.Match) -> str:
        commentary = code_commentary(match.group(1))
        if not commentary:
            return "[code block removed: no comments or docstrings]"
        return f"[code block removed; its comments and docstrings:]\n{commentary}\n[end of code block]"
    return CODE_BLOCK.sub(replace, reply).strip()


def write_strided_batches(items: list[BlindItem], n_batches: int, batch_dir: Path) -> None:
    """Item i goes to batch i mod n_batches, as "## ITEM <id>" sections in batch_<NN>.md."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_batches):
        body = "\n\n".join(f"## ITEM {item['item']}\n\n{item['text'] or '[empty reply]'}"
                           for item in items[index::n_batches])
        (batch_dir / f"batch_{index + 1:02d}.md").write_text(body + "\n")


def pack_batches(rendered_items: list[str], batch_chars: int) -> list[list[str]]:
    """Consecutive items packed into batches of at most batch_chars characters (a larger item is a batch of its own)."""
    batches: list[list[str]] = []
    for rendered in rendered_items:
        if batches and sum(map(len, batches[-1])) + len(rendered) <= batch_chars:
            batches[-1].append(rendered)
        else:
            batches.append([rendered])
    return batches


def turn_outcome(attempt: CheckAttempt) -> str:
    if not attempt.get("answer"):
        return "[HARNESS] This turn submitted no code."
    if attempt.get("success"):
        return "[HARNESS] The submitted code PASSED the visible tests; the episode ended."
    return "[HARNESS] The submitted code FAILED the visible tests and was rejected."


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_label_rows[Row: LabelRow](label_set_dir: Path, row_type: type[Row]) -> list[Row]:
    """Every labeler output row in <dir>/labels/batch_*.jsonl, taken to be a row_type (not validated)."""
    return [cast(Row, json.loads(line)) for path in sorted((label_set_dir / "labels").glob("batch_*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def labelled_pairs[Key: Mapping[str, object], Row: LabelRow](key_by_item: Mapping[str, Key],
                                                              label_row_by_item: Mapping[str, Row]
                                                              ) -> list[tuple[Key, Row]]:
    """(key, label row) of every labeled item, in key order."""
    return [(key_by_item[item], label_row_by_item[item]) for item in key_by_item if item in label_row_by_item]


def write_joined_and_tally(label_set_dir: Path, key_by_item: Mapping[str, Mapping[str, object]],
                           label_row_by_item: Mapping[str, LabelRow], quote_problems: Sequence[Mapping[str, object]],
                           counts_by_group: dict[str, dict[str, Counter]]) -> None:
    """<dir>/labels_joined.jsonl (key and label per item) and <dir>/tally.json (coverage, quote problems, counts)."""
    joined = [{**key, **label_row} for key, label_row in labelled_pairs(key_by_item, label_row_by_item)]
    (label_set_dir / "labels_joined.jsonl").write_text("".join(json.dumps(row) + "\n" for row in joined))
    tally: LabelTally = {
        "n_items": len(key_by_item), "n_labeled": len(joined),
        "unlabeled_items": sorted(set(key_by_item) - set(label_row_by_item)),
        "unknown_items": sorted(set(label_row_by_item) - set(key_by_item)),
        "quote_problems": quote_problems,
        "counts_by_group": {group: {field: dict(counter.most_common()) for field, counter in counters.items()}
                            for group, counters in sorted(counts_by_group.items())},
    }
    (label_set_dir / "tally.json").write_text(json.dumps(tally, indent=1))
