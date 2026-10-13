"""Mean response-token hidden states for the same-prompt pairs, on gpt-oss-120b.

This follows the Assistant Axis paper (arXiv:2601.10387), section 2.1.2:
the vector for a reply is the mean of the post-MLP residual stream over
every token of that reply, at the middle layer.

gpt-oss-120b has 36 layers, indexed from 0. The middle layer is 18, the
same index this repo uses for that model. The forward pass is the user
message and the reply stored in the pair file.

Launched by slurm/extract_response_mean.sbatch. From the repo root:
    python experiments/reward-hack-axis/scripts/extract_response_mean.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from selfconcept.assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel
from selfconcept.assistant_axis.models import get_config

MODEL = "openai/gpt-oss-120b"
MAX_LENGTH = 8192
ATTENTION = "kernels-community/vllm-flash-attn3"


def effort(row: dict) -> str:
    return "low" if row["reasoning_effort"] == "low" else "medium"


def assistant_span(spans: list[dict]) -> dict:
    assistant = [span for span in spans if span["role"] == "assistant"]
    if len(assistant) != 1 or assistant[0]["n_tokens"] < 1:
        raise SystemExit("conversation does not have one assistant reply")
    return assistant[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attn-implementation", default=ATTENTION)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    pairs_path = repo / "experiments/reward-hack-axis/data/same_prompt_pairs.jsonl"
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if len(rows) != 68:
        raise SystemExit(f"expected 68 replies, found {len(rows)}")

    layer = get_config(args.model)["target_layer"]
    model = ProbingModel(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    encoder = ConversationEncoder(model.tokenizer, model.model_name)
    extractor = ActivationExtractor(model, encoder)
    n_layers = len(model.get_layers())
    if n_layers != 36 or layer != 18:
        raise SystemExit(f"expected middle layer 18 of 36, found layer {layer} of {n_layers}")

    vectors: list[torch.Tensor] = []
    meta: list[dict] = []
    for row in rows:
        chat_kwargs = {"reasoning_effort": effort(row)}
        activations, batch_meta = extractor.batch_conversations(
            [row["messages"]], layer=[layer], max_length=MAX_LENGTH, **chat_kwargs
        )
        _, spans, _ = encoder.build_batch_turn_spans([row["messages"]], **chat_kwargs)
        span = assistant_span(spans)
        n_tokens = batch_meta["actual_lengths"][0]
        if n_tokens > MAX_LENGTH or batch_meta["truncated_lengths"][0] != n_tokens or span["end"] > n_tokens:
            raise SystemExit(f"{row['id']} was truncated ({n_tokens} tokens)")
        reply = activations[:, 0, span["start"]:span["end"], :]
        vectors.append(reply.mean(dim=1).to(dtype=torch.float32).cpu())
        meta.append({
            "id": row["id"],
            "pair_id": row["pair_id"],
            "task_id": row["task_id"],
            "split": row["split"],
            "condition": row["condition"],
            "hack_type": row["hack_type"],
            "n_tokens": n_tokens,
            "n_response_tokens": span["end"] - span["start"],
        })
        del activations

    stacked = torch.stack(vectors)
    if stacked.shape != (len(rows), 1, model.hidden_size):
        raise SystemExit(f"unexpected activation shape {tuple(stacked.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "response_mean.pt"
    torch.save(
        {
            "model": args.model,
            "layers": [layer],
            "position": "mean of every token in the assistant reply",
            "activations": stacked,
            "rows": meta,
        },
        out,
    )
    summary = {
        "rows": len(rows),
        "layer": layer,
        "hidden_size": model.hidden_size,
        "max_tokens": max(item["n_tokens"] for item in meta),
        "output": str(out),
    }
    out.with_name("response_mean_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
