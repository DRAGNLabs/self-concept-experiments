"""Measure paired endpoint gaps under existing SOO interventions.

Run `python -m selfconcept.soo.measure_overlap --help` for arguments.
Input JSONL rows require id, family, split, kind, self_prompt, other_prompt.
See experiments/soo/OVERLAP.md for the input contract and interpretation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import torch

from selfconcept.common.chat import chat_template_kwargs
from .activations import get_decoder_layers
from .overlap import last_valid_indices, measure_condition, summarize
from .steering import (
    POSITION_MODES, PositionalSteering, apply_steering, get_vector,
    load_vectors, random_matched_vector, response_marker,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"artifact does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return {str(p): file_hash(p) for p in files}


def read_pairs(path, split):
    """Validate IDs and declared family separation before selecting a split."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids, families, prompts = set(), {}, {}
    required = ("id", "family", "split", "kind", "self_prompt", "other_prompt")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError(f"every pair needs nonempty strings: {required}")
        if row["id"] in ids:
            raise ValueError(f"duplicate pair ID: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in ("fit", "development", "final"):
            raise ValueError("split must be fit, development, or final")
        if row["kind"] not in ("self_other", "nonsocial"):
            raise ValueError("kind must be self_other or nonsocial")
        family, assigned = row["family"], row["split"]
        if families.setdefault(family, assigned) != assigned:
            raise ValueError(f"family crosses splits: {family}")
        for key in ("self_prompt", "other_prompt"):
            # Whitespace differences must not disguise literal leakage.
            prompt = " ".join(row[key].split())
            if prompts.setdefault(prompt, assigned) != assigned:
                raise ValueError("a prompt occurs in more than one split")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no pairs in split {split}")
    return rows, selected


def encode_pairs(tokenizer, pairs, batch_size, device, chat_kwargs):
    """Render once and reuse the exact token tensors in every condition."""
    batches, token_records = [], []
    for start in range(0, len(pairs), batch_size):
        group = pairs[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row[key]}], tokenize=False,
            add_generation_prompt=True, **chat_kwargs,
        ) for row in group for key in ("self_prompt", "other_prompt")]
        enc = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        last = last_valid_indices(enc["attention_mask"])
        for j, text in enumerate(texts):
            ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
            token = int(enc["input_ids"][j, last[j]])
            token_records.append({
                "id": group[j // 2]["id"], "member": "self" if j % 2 == 0 else "other",
                "rendered_prompt": text, "input_ids": ids, "n_tokens": len(ids),
                "endpoint_token_id": token,
                "endpoint_token": tokenizer.convert_ids_to_tokens(token),
                "endpoint_decoded": tokenizer.decode([token]),
            })
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches, token_records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", help="model/tokenizer revision; resolved commit is recorded")
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--split", choices=("development", "final"), default="development")
    p.add_argument("--out", type=Path, required=True, help="new run directory; must not exist")
    p.add_argument("--layer", type=int, required=True, help="zero-based intervention layer")
    p.add_argument("--residual-layers", type=int, nargs="+", help="default: intervention block and every later block")
    p.add_argument("--vectors", type=Path)
    p.add_argument("--token-mode", choices=("last", "mean"), default="last")
    p.add_argument("--mode", choices=("add", "project"), default="add")
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--positions", choices=POSITION_MODES, default="all")
    p.add_argument("--random-seeds", type=int, nargs="*", default=[])
    p.add_argument("--adapter", type=Path, help="local PEFT adapter; compared separately to base")
    p.add_argument("--batch-size", type=int, default=4, help="pairs per batch")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc.")
    p.add_argument("--device-map", help="e.g. auto; overrides --device")
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap/reproducibility seed")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing run: {args.out}")
    if args.batch_size < 1 or args.bootstrap < 1:
        raise ValueError("batch size and bootstrap count must be positive")
    if args.random_seeds and not args.vectors:
        raise ValueError("random controls require --vectors")
    if not torch.isfinite(torch.tensor(args.alpha)):
        raise ValueError("alpha must be finite")
    all_pairs, pairs = read_pairs(args.probes, args.split)
    vector = None
    hashes = {str(args.probes): file_hash(args.probes)}
    if args.vectors:
        hashes.update(artifact_hashes(args.vectors))
        data = load_vectors(args.vectors)
        if data.get("model") != args.model:
            raise ValueError("vector's model ID does not match --model")
        if args.layer < 0:
            raise ValueError("layer must be nonnegative")
        vector = get_vector(data, args.layer, args.token_mode)
        if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("steering vector must be finite, nonzero, and one-dimensional")
    if args.adapter:
        hashes.update(artifact_hashes(args.adapter))
    if Path(args.model).is_dir():
        hashes.update(artifact_hashes(args.model))

    from transformers import AutoTokenizer
    from selfconcept.common.loading import load_causal_lm

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer needs a padding or EOS token")
    tokenizer.padding_side = "right"
    kwargs = {"dtype": getattr(torch, args.dtype)}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = load_causal_lm(args.model, **kwargs)
    if not args.device_map:
        model.to(args.device)
    device = model.get_input_embeddings().weight.device
    chat_kwargs = chat_template_kwargs()
    batches, token_records = encode_pairs(tokenizer, pairs, args.batch_size, device, chat_kwargs)
    marker = response_marker(tokenizer, chat_kwargs) if args.positions != "all" else None
    if marker:
        # Refuse an ambiguous positional comparison instead of silently using a fallback.
        for row in token_records:
            if row["input_ids"][-len(marker):] != marker:
                raise ValueError("rendered prompt does not end in the expected response marker")

    conditions = {}

    def run(name, intervention=None):
        print(f"Measuring {name}: {len(pairs)} pairs", flush=True)
        conditions[name] = measure_condition(model, batches, args.layer, args.residual_layers, intervention)

    def steering(v, alpha):
        return lambda: apply_steering(
            model, args.layer, v, alpha, args.mode,
            PositionalSteering(marker, args.positions) if marker else None,
        )

    run("base")
    if vector is not None:
        if vector.numel() != conditions["base"]["hook_before"].shape[-1]:
            raise ValueError("vector width does not match the intervention site")
        run("inactive", steering(vector, 0.))
        run(args.mode, steering(vector, args.alpha))
        for seed in sorted(set(args.random_seeds)):
            run(f"random_s{seed}", steering(random_matched_vector(vector, seed), args.alpha))
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), torch_device=str(device))
        run("adapter")

    summary, records = summarize(conditions, pairs, args.bootstrap, args.seed)
    root = Path(__file__).resolve().parents[3]
    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    source_files = [Path(__file__), Path(__file__).with_name("overlap.py"),
                    Path(__file__).with_name("steering.py"), Path(__file__).with_name("activations.py"),
                    root / "src/selfconcept/common/loading.py", root / "src/selfconcept/common/chat.py"]
    modules = dict(model.named_modules())
    def module_path(target):
        return next(name for name, module in modules.items() if module is target)
    from .activations import attn_out_proj
    from .overlap import decoder_final_norm
    metadata = {
        "schema_version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "command": sys.argv,
        "model_resolved_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "model_config": model.config.to_dict(),
        "python": sys.version,
        "device_map": getattr(model, "hf_device_map", None),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
        "code_commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--short"),
        "source_sha256": {str(p.relative_to(root)): file_hash(p) for p in source_files},
        "input_sha256": hashes, "pairs": pairs,
        "declared_split_ids": {split: [p["id"] for p in all_pairs if p["split"] == split]
                               for split in ("fit", "development", "final")},
        "split_validation": "IDs, literal prompt overlap and declared families checked within this input only; artifact training membership is not verified",
        "chat_kwargs": chat_kwargs, "chat_template": tokenizer.chat_template,
        "padding_side": tokenizer.padding_side, "response_marker_ids": marker,
        "capture_modules": {"hook": module_path(attn_out_proj(get_decoder_layers(model)[args.layer])),
                            "final_norm": module_path(decoder_final_norm(model)),
                            **{site: module_path(get_decoder_layers(model)[int(site.removeprefix("residual_L"))])
                               for site in conditions["base"] if site.startswith("residual_L")}},
        "conditions": list(conditions), "dropout": "disabled via eval", "generated_tokens": 0,
        "metric": "float32 mean squared paired difference over activation dimensions, then pairs",
        "ci": "95% percentile bootstrap of whole declared families; pair-weighted mean; null with fewer than 2 families",
        "adapter_hook_before": "adapter-active module output, not the unadapted base; use condition base for that comparison",
    }
    # A new directory and a manifest written last distinguish a complete run.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.out / "tokens.jsonl").write_text("".join(json.dumps(row) + "\n" for row in token_records))
    torch.save(conditions, args.out / "activations.pt")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
