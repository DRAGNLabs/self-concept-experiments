"""Freeze neutral, article-disjoint probe text from the already-cached WikiText."""
import hashlib
import json
from pathlib import Path
import random
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "experiments/cotness/data/neutral.jsonl"
CACHE = Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"


def main():
    rows = []
    seen = set()
    for source_split, split, n in [("train", "train", 96), ("validation", "validation", 24), ("test", "test", 24)]:
        path = CACHE / f"{source_split}-00000-of-00001.parquet"
        articles, title = {}, None
        for line in pd.read_parquet(path)["text"]:
            line = line.strip()
            if re.fullmatch(r"= [^=]+ =", line):
                title = line
            elif title and len(line) >= 500 and not line.startswith("="):
                articles.setdefault(title, line)
        candidates = sorted(articles.items())
        random.Random(1729).shuffle(candidates)
        selected = 0
        for title, text in candidates:
            # Both article title and content must be unseen across splits.
            digest = hashlib.sha256(text.encode()).hexdigest()
            if title in seen or digest in seen:
                continue
            seen.update((title, digest))
            rows.append({"id": f"wikitext:{source_split}:{digest[:16]}", "split": split,
                         "article": title, "text": text, "source_file": path.name,
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            selected += 1
            if selected == n:
                break
        if selected != n:
            raise ValueError(f"Only {selected} qualifying articles for {split}")
    serialized = "".join(json.dumps(r) + "\n" for r in rows)
    if DEST.exists() and DEST.read_text() != serialized:
        raise RuntimeError("Refusing to replace a different frozen probe corpus")
    DEST.write_text(serialized)
    print(f"{DEST}: {len(rows)} documents; sha256={hashlib.sha256(serialized.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
