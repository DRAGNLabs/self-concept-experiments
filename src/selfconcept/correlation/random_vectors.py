"""Cache region-mean residuals for many random-direction controls in one forward pass.

Uses the existing transcript tokenizer, token alignment and post-decoder hook.
Projection of a mean equals the mean token projection. No generation is performed.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from selfconcept.common.jsonl import append_jsonl, read_jsonl
from selfconcept.common.loading import load_causal_lm
from selfconcept.measurement.capture import capture
from selfconcept.measurement.templates import response_spans
from selfconcept.soo.activations import get_decoder_layers
from .generate import check_context_budget
from .transcripts import tokenize_transcript, transcript_token_indices_by_region

REGIONS = ("prompt", "cot", "final")
SCHEMA = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def random_directions(hidden_size, count=256, seed=1729, layer=17):
    """Independent isotropic unit vectors; layer-specific, prefix-stable RNG."""
    if hidden_size < 1 or count < 1 or seed < 0 or layer < 0:
        raise ValueError("Invalid direction dimensions or seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, layer]))
    directions = rng.standard_normal((count, hidden_size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions.astype(np.float32)


def pooled_scorer(indices, sequence_length):
    """Pool only the same processed token indices used by transcript scoring."""
    selected = {region: [i for i in indices.get(region, []) if i < sequence_length - 1]
                for region in REGIONS}
    def score(*, layer, residual):
        if len(residual) != sequence_length:
            raise ValueError("Unexpected forward token alignment")
        return torch.stack([residual[selected[region]].mean(0) if selected[region]
                            else residual.new_full((residual.shape[-1],), float("nan"))
                            for region in REGIONS])
    return score, [len(selected[r]) for r in REGIONS]


def score_one(model, tokenizer, transcript, layers, out):
    tokenized = tokenize_transcript(tokenizer, transcript)
    ids = tokenized.prompt_ids + tokenized.response_ids
    check_context_budget(model, len(ids), 0)
    spans = response_spans(tokenized.prompt, transcript["raw_response"], "gpt-oss")
    indices, alignment = transcript_token_indices_by_region(
        tokenizer, transcript["messages"], transcript["raw_response"], tokenized, spans)
    if alignment != "exact":
        raise ValueError("Cannot align response tokens exactly")
    pool, counts = pooled_scorer(indices, len(ids))
    inputs = torch.tensor([ids], device=next(model.get_input_embeddings().parameters()).device)
    with torch.inference_mode(), capture(model, layers, pool) as values:
        model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
    if any(len(chunks) != 1 for chunks in values.values()):
        raise ValueError("Expected one residual capture per layer")
    key = hashlib.sha256(json.dumps([transcript["scenario"], transcript["example_id"]]).encode()).hexdigest()
    relative = f"residuals/{key}.npz"
    path = out / relative
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{f"layer_{layer}": chunks[0].numpy() for layer, chunks in values.items()})
    temporary.replace(path)
    return {"example_id": transcript["example_id"], "scenario": transcript["scenario"],
            "turn": transcript["turn"], "outcome": transcript["outcome"], "status": "complete",
            "residuals": relative, "region_counts": counts, "prompt_tokens": len(tokenized.prompt_ids),
            "response_tokens": len(tokenized.response_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--revision", required=True, help="Pinned checkpoint commit")
    parser.add_argument("--layers", nargs="+", type=int, default=[17])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke test only; recorded in manifest")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        parser.error("Invalid shard or layers")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "residuals").mkdir(exist_ok=True)
    rows = read_jsonl(args.transcripts)
    if any(row["outcome"]["model_key"] != args.model for row in rows):
        raise ValueError("Transcript model differs from scoring model")
    if len({(r["scenario"], r["example_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate transcript identifiers")
    rows = rows[args.shard::args.shards]
    if args.limit is not None:
        rows = rows[:args.limit]
    identity = {"schema": SCHEMA, "model": args.model, "revision": args.revision,
                "transcripts_sha256": sha256(args.transcripts), "layers": args.layers,
                "shard": args.shard, "shards": args.shards, "limit": args.limit,
                "n_expected": len(rows), "torch": torch.__version__, "transformers": transformers.__version__,
                "runner_sha256": sha256(__file__)}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError("Refusing to mix residual-cache configurations")
    manifest_path.write_text(json.dumps({"identity": identity, "stage": "started"}, indent=2))
    path = args.out / "records.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    done = {(r["scenario"], r["example_id"]) for r in previous if r["status"] == "complete"}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = load_causal_lm(args.model, revision=args.revision, local_files_only=True,
                          dtype=torch.bfloat16, device_map="auto",
                          attn_implementation="kernels-community/vllm-flash-attn3")
    model.eval()
    num_layers = len(get_decoder_layers(model))
    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    if max(args.layers) >= num_layers:
        raise ValueError("Requested layer is outside model")
    errors = 0
    for i, row in enumerate(rows):
        if (row["scenario"], row["example_id"]) in done:
            continue
        try:
            result = score_one(model, tokenizer, row, args.layers, args.out)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors += 1
            result = {"example_id": row["example_id"], "scenario": row["scenario"], "status": "error_oom"}
        append_jsonl(path, result)
        logging.info("%s/%s %s %s", i + 1, len(rows), result["status"], row["example_id"])
    manifest_path.write_text(json.dumps({"identity": identity, "stage": "complete", "errors": errors,
                                        "num_layers": num_layers, "hidden_size": hidden_size}, indent=2))
    if args.limit is not None and errors:
        raise RuntimeError("Smoke test had OOM errors; refusing to release dependent scoring jobs")


if __name__ == "__main__":
    main()
