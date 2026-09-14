"""Shared model / intervention / decoding setup for the eval harnesses.

Mirrors the inline block in evaluate.py (which predates this module and was
left untouched mid-study so its numbers stay byte-comparable): same flags,
same load order (base -> optional PEFT adapter -> device), same o_proj
steering hook and norm-matched random control, same per-example sampling
seed. New harnesses should use this instead of copying evaluate.py.
"""

import argparse
import zlib
from pathlib import Path

import torch

from .loading import load_causal_lm


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="optional PEFT adapter path (SOO fine-tuned)")
    parser.add_argument("--quant-4bit", action="store_true", help="load base model in 4-bit (paper setup)")
    parser.add_argument("--steer-vectors", type=Path, help="steering vectors .pt from scripts/extract_steering.py")
    parser.add_argument("--steer-layer", type=int, help="decoder layer to steer at (required with --steer-vectors)")
    parser.add_argument("--steer-alpha", type=float, default=1.0, help="steering strength (1.0 = full mean self-other difference)")
    parser.add_argument("--steer-mode", choices=["add", "project"], default="add")
    parser.add_argument("--steer-token-mode", choices=["last", "mean"], default="last", help="which extraction convention's vector to use")
    parser.add_argument("--steer-random-seed", type=int, help="control: replace the vector with a random one of matched norm")
    parser.add_argument(
        "--device-map",
        help="pass device_map to from_pretrained (e.g. 'auto' to shard models "
        "too big for one GPU, like Llama-2-70b); skips the single-device .to()",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (default). >0 samples with pure temperature scaling "
        "(top_p/top_k disabled so T is the only knob), seeded per example",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="with --temperature: RNG seed, mixed with each example_id so "
        "results are reproducible and independent of subset/order",
    )


def load_model(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None):
    """Load tokenizer + model per the args; returns (model, tokenizer, device, steering).

    `steering` is the summary dict written into result files (None when not
    steering). The steering hook, if any, is registered for the model's
    lifetime, so it applies to every generate() call including multi-turn loops.
    """
    from transformers import AutoTokenizer

    device = pick_device()
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.quant_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = load_causal_lm(args.model, quantization_config=bnb, dtype=dtype)
    elif args.device_map:
        model = load_causal_lm(args.model, dtype=dtype, device_map=args.device_map)
    else:
        model = load_causal_lm(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    if not args.quant_4bit and not args.device_map:
        model.to(device)
    model.eval()

    steering = None
    if args.steer_vectors:
        if args.steer_layer is None:
            (parser.error if parser else _fail)("--steer-vectors requires --steer-layer")
        from .steering import get_vector, load_vectors, random_matched_vector, steer_o_proj

        vector = get_vector(load_vectors(args.steer_vectors), args.steer_layer, args.steer_token_mode)
        if args.steer_random_seed is not None:
            vector = random_matched_vector(vector, args.steer_random_seed)
        steer_o_proj(model, args.steer_layer, vector, args.steer_alpha, args.steer_mode)
        steering = {
            "vectors": str(args.steer_vectors),
            "layer": args.steer_layer,
            "alpha": args.steer_alpha,
            "mode": args.steer_mode,
            "token_mode": args.steer_token_mode,
            "random_seed": args.steer_random_seed,
            "vector_norm": round(vector.norm().item(), 6),
        }
        print(f"Steering: {steering}")
    return model, tokenizer, device, steering


def _fail(message: str) -> None:
    raise SystemExit(message)


def sampling_kwargs(args: argparse.Namespace, example_id: str, turn: int = 0) -> dict:
    """generate() decoding kwargs; seeds torch per (seed, example, turn) when sampling."""
    if args.temperature > 0:
        key = f"{args.sample_seed}:{example_id}" + (f":{turn}" if turn else "")
        torch.manual_seed(zlib.crc32(key.encode()))
        return {"do_sample": True, "temperature": args.temperature, "top_p": 1.0, "top_k": 0}
    return {"do_sample": False}
