"""Validate and save native-tokenizer spans; loads no model weights.

Run from the repository root with PYTHONPATH=src and USE_TORCH=0 if desired.
Reads the cached tokenizer only. Output must be a new directory.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from selfconcept.soo.goals_roles import tokenize_example
from selfconcept.soo.score_goals_roles import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    parser.add_argument("--chat-kwargs", default='{"enable_thinking": false}')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    rows = [r for path in args.data for r in read_jsonl(path)]
    if not rows or len({r["example_id"] for r in rows}) != len(rows):
        parser.error("expected a nonempty dataset with unique IDs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    kwargs = json.loads(args.chat_kwargs)
    mappings = [tokenize_example(r, tokenizer, kwargs) for r in rows]
    lengths = [len(r["input_ids"]) for r in mappings]
    summary = {"status": "passed", "model": args.model, "revision": args.revision,
               "chat_kwargs": kwargs, "prompts_checked": len(rows),
               "min_tokens": min(lengths), "max_tokens": max(lengths),
               "versions": {p: version(p) for p in ("transformers", "tokenizers")},
               "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.data},
               "model_weights_loaded": False}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "token_spans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mappings))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
