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


def chat_kwargs_for(row: dict) -> dict:
    if row.get("chat_kwargs"):
        return dict(row["chat_kwargs"])
    return {"reasoning_effort": effort(row)}


def load_probing(model_name: str, revision: str | None, attn_implementation: str | None) -> ProbingModel:
    if revision is None:
        return ProbingModel(model_name, dtype=torch.bfloat16, attn_implementation=attn_implementation)
    from transformers import AutoTokenizer

    from selfconcept.common.loading import load_causal_lm

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    loaded = load_causal_lm(model_name, revision=revision, **load_kwargs)
    loaded.eval()
    return ProbingModel.from_existing(loaded, tokenizer, model_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--attn-implementation", default=ATTENTION)
    parser.add_argument("--write-axis", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    pairs_path = args.pairs or (repo / "experiments/reward-hack-axis/data/same_prompt_pairs.jsonl")
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if args.pairs is None and len(rows) != 68:
        raise SystemExit(f"expected 68 replies, found {len(rows)}")
    if not rows:
        raise SystemExit(f"no rows in {pairs_path}")

    layer = 26 if "Muse-Glimmer" in args.model else get_config(args.model)["target_layer"]
    model = load_probing(args.model, args.revision, args.attn_implementation)
    encoder = ConversationEncoder(model.tokenizer, model.model_name)
    extractor = ActivationExtractor(model, encoder)
    n_layers = len(model.get_layers())
    if args.model == MODEL and (n_layers != 36 or layer != 18):
        raise SystemExit(f"expected middle layer 18 of 36, found layer {layer} of {n_layers}")
    if "Muse-Glimmer" in args.model and (n_layers != 52 or layer != 26):
        raise SystemExit(f"expected middle layer 26 of 52, found layer {layer} of {n_layers}")

    vectors: list[torch.Tensor] = []
    meta: list[dict] = []
    for row in rows:
        chat_kwargs = chat_kwargs_for(row)
        activations, batch_meta = extractor.batch_conversations(
            [row["messages"]], layer=[layer], max_length=MAX_LENGTH, include_thinking=False, **chat_kwargs
        )
        _, spans, _ = encoder.build_batch_turn_spans([row["messages"]], include_thinking=False, **chat_kwargs)
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
        "pairs": str(pairs_path),
        "revision": args.revision,
        "output": str(out),
    }
    if args.write_axis:
        train_hack = [stacked[i, 0] for i, row in enumerate(meta) if row["split"] == "train" and row["condition"] == "hack"]
        train_honest = [stacked[i, 0] for i, row in enumerate(meta) if row["split"] == "train" and row["condition"] == "honest"]
        if not train_hack or not train_honest:
            raise SystemExit("train split needs both a hack reply and an honest reply")
        axis = torch.stack(train_hack).mean(dim=0) - torch.stack(train_honest).mean(dim=0)
        axis_path = args.output_dir / "hack_axis.pt"
        torch.save(
            {
                "axis": axis,
                "layer": layer,
                "formula": "mean(train hack) - mean(train honest)",
                "model": args.model,
                "revision": args.revision,
                "pairs": str(pairs_path),
                "train_pairs": len(train_hack),
            },
            axis_path,
        )
        summary["train_pairs"] = len(train_hack)
        summary["holdout_pairs"] = sum(1 for row in meta if row["split"] == "holdout" and row["condition"] == "hack")
        summary["axis"] = str(axis_path)
    out.with_name("response_mean_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
