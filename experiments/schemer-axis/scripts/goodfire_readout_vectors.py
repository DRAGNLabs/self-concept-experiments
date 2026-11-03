"""All-layer activations of Goodfire's synthetic reward-hacking conversations (HF Goodfire/reward-hack-data,
synthetic_data/dataset.jsonl) at three readouts:

- close: the closing markup of each readout's tool call through its end token, averaged (their choice);
- last: that tool call's end token;
- all_response: every assistant tool-call and response token of the conversation, averaged.

close and last give one row per (conversation, readout); all_response one row per conversation.
Writes <output-dir>/goodfire_readouts.pt; vectors are kept in bf16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import BFloat16

from selfconcept.assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel
from selfconcept.assistant_axis.internals.model_specifics.message_spans import ReadoutSpecifics
from selfconcept.assistant_axis.internals.readouts import (
    ToolCallingMessage, all_response_positions, last_token_positions, readout_specifics_for, tool_call_close_positions,
    tool_calling_conversation_ids,
)
from selfconcept.common.hf_strong_types import HFTokenizer

class GoodfireRow(TypedDict):
    id: str
    group_id: str
    split: str
    metadata: dict[str, str]
    messages: list[ToolCallingMessage]
    template_kwargs: dict[str, bool]
    readouts: list[dict]


class ReadoutRow(TypedDict):
    id: str
    group_id: str
    split: str
    dataset: str
    cell: str
    readout: str
    labels: dict[str, int]


class RowPositions(TypedDict):
    close: list[list[int]]
    last: list[list[int]]
    all_response: list[int]


def row_positions(specifics: ReadoutSpecifics, tokenizer: HFTokenizer, full_ids: list[int], row: GoodfireRow) -> RowPositions:
    message_indices = [readout["selector"]["message_index"] for readout in row["readouts"]]
    return {
        "close": [tool_call_close_positions(specifics, tokenizer, full_ids, row["messages"], index) for index in message_indices],
        "last": [last_token_positions(specifics, tokenizer, full_ids, row["messages"], index) for index in message_indices],
        "all_response": all_response_positions(specifics, tokenizer, full_ids, row["messages"]),
    }


def mean_at(activations: BFloat16[torch.Tensor, "layer seq hidden"], positions: list[int]) -> BFloat16[torch.Tensor, "layer hidden"]:
    return activations[:, positions, :].float().mean(dim=1).to(torch.bfloat16).cpu()


def locate_readouts(
    specifics: ReadoutSpecifics, tokenizer: HFTokenizer, rows: list[GoodfireRow], chat_kwargs: dict[str, str]
) -> tuple[list[list[int]], list[RowPositions]]:
    full_ids_per_row = [
        tool_calling_conversation_ids(tokenizer, row["messages"], **row["template_kwargs"], **chat_kwargs)
        for row in rows
    ]
    positions_per_row = [row_positions(specifics, tokenizer, full_ids, row) for row, full_ids in zip(rows, full_ids_per_row)]
    print(f"located readouts in {len(rows)} conversations; longest {max(map(len, full_ids_per_row))} tokens")
    return full_ids_per_row, positions_per_row


def extract_readouts(
    extractor: ActivationExtractor, full_ids_per_row: list[list[int]], positions_per_row: list[RowPositions], batch_size: int
) -> dict[str, list[torch.Tensor]]:
    vectors_by_readout: dict[str, list[torch.Tensor]] = {"close": [], "last": [], "all_response": []}
    for batch_start in range(0, len(full_ids_per_row), batch_size):
        batch_ids = full_ids_per_row[batch_start:batch_start + batch_size]
        activations, _ = extractor.batch_token_ids(batch_ids, layer=None, max_length=max(map(len, batch_ids)))
        for offset, positions in enumerate(positions_per_row[batch_start:batch_start + batch_size]):
            conversation_activations = activations[:, offset]
            vectors_by_readout["close"].extend(mean_at(conversation_activations, readout) for readout in positions["close"])
            vectors_by_readout["last"].extend(mean_at(conversation_activations, readout) for readout in positions["last"])
            vectors_by_readout["all_response"].append(mean_at(conversation_activations, positions["all_response"]))
        del activations
    return vectors_by_readout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chat-kwargs", type=json.loads, default={}, help='e.g. {"reasoning_effort": "low"}')
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=16, help="4 fits Muse-Glimmer-30B on an 80 GB GPU")
    parser.add_argument("--limit", type=int, help="first N conversations only (smoke runs)")
    parser.add_argument("--positions-only", action="store_true", help="locate every readout without loading the model")
    args = parser.parse_args()

    rows: list[GoodfireRow] = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()][: args.limit]
    specifics = readout_specifics_for(args.model)
    if args.positions_only:
        from transformers import AutoTokenizer
        locate_readouts(specifics, AutoTokenizer.from_pretrained(args.model), rows, args.chat_kwargs)
        return
    model = ProbingModel(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    assert model.tokenizer is not None
    full_ids_per_row, positions_per_row = locate_readouts(specifics, model.tokenizer, rows, args.chat_kwargs)
    extractor = ActivationExtractor(model, ConversationEncoder(model.tokenizer, model.model_name))
    vectors_by_readout = extract_readouts(extractor, full_ids_per_row, positions_per_row, args.batch_size)

    readout_rows: list[ReadoutRow] = [
        {"id": row["id"], "group_id": row["group_id"], "split": row["split"], "dataset": row["metadata"]["dataset"],
         "cell": row["metadata"]["cell"], "readout": readout["name"], "labels": readout["labels"]}
        for row in rows for readout in row["readouts"]
    ]
    conversation_rows = [
        {"id": row["id"], "group_id": row["group_id"], "split": row["split"], "dataset": row["metadata"]["dataset"],
         "cell": row["metadata"]["cell"]}
        for row in rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "goodfire_readouts.pt"
    torch.save({
        "model": args.model, "dataset": str(args.dataset), "chat_kwargs": args.chat_kwargs,
        "readout_rows": readout_rows, "conversation_rows": conversation_rows,
        "close": torch.stack(vectors_by_readout["close"]), "last": torch.stack(vectors_by_readout["last"]),
        "all_response": torch.stack(vectors_by_readout["all_response"]),
    }, out)
    print(f"{out}: close/last {tuple(torch.stack(vectors_by_readout['close']).shape)}, all_response {len(conversation_rows)} rows")


if __name__ == "__main__":
    main()
