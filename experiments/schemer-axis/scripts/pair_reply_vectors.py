"""All-layer reply-mean vectors for Chris's same-prompt hack/honest pair files (reward-hack-axis on origin/Chris).

Same recipe as his extract_response_mean.py: one forward pass over the user message plus the stored reply,
thinking excluded, mean over the reply tokens. It keeps every layer instead of only layer 18, so the axis
can be compared with the persona axes layer by layer. Writes {rows, activations (n_rows, n_layers, hidden)}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from selfconcept.assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel

MAX_LENGTH = 8192
ATTENTION = "kernels-community/vllm-flash-attn3"


def chat_kwargs_for(row: dict) -> dict:
    if row.get("chat_kwargs"):
        return dict(row["chat_kwargs"])
    return {"reasoning_effort": "low" if row["reasoning_effort"] == "low" else "medium"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = ProbingModel(args.model, dtype=torch.bfloat16, attn_implementation=ATTENTION)
    assert model.tokenizer is not None
    encoder = ConversationEncoder(model.tokenizer, model.model_name)
    extractor = ActivationExtractor(model, encoder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pairs_path in args.pairs:
        rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
        vectors: list[torch.Tensor] = []
        for row in rows:
            chat_kwargs = chat_kwargs_for(row)
            activations, batch_meta = extractor.batch_conversations(
                [row["messages"]], layer=None, max_length=MAX_LENGTH, include_thinking=False, **chat_kwargs
            )
            _, spans, _ = encoder.build_batch_turn_spans([row["messages"]], include_thinking=False, **chat_kwargs)
            (span,) = [span for span in spans if span["role"] == "assistant"]
            if batch_meta["actual_lengths"][0] > MAX_LENGTH or span["end"] > batch_meta["actual_lengths"][0]:
                raise SystemExit(f"{row['id']} was truncated")
            vectors.append(activations[:, 0, span["start"]:span["end"], :].mean(dim=1).float().cpu())
            del activations
        meta = [{key: row[key] for key in ("id", "pair_id", "task_id", "split", "condition", "hack_type")} for row in rows]
        out = args.output_dir / f"{pairs_path.stem}.pt"
        torch.save({"model": args.model, "pairs": str(pairs_path), "rows": meta, "activations": torch.stack(vectors)}, out)
        print(f"{out}: {tuple(torch.stack(vectors).shape)}")


if __name__ == "__main__":
    main()
