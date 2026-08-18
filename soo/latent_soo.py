"""Latent SOO: layer-wise MSE between self- and other-referencing activations
on the 52 probe pairs (paper section 3.1.1 / A.1.3, Table 4).

Usage:
    python -m soo.latent_soo --model allenai/OLMo-2-0425-1B-Instruct \
        [--adapter results/checkpoints/olmo2-1b/seed0] [--tag baseline_1b]

Captures the o_proj output of every decoder layer for each pair (prompts
right-padded to a common length per pair) and reports per-layer mean MSE plus
the overall mean, saved to results/latent/.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activations import get_decoder_layers
from .evaluate import pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--probes", type=Path, default=Path("data/latent_probes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/latent"))
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    device = pick_device()
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    layers = get_decoder_layers(model)
    stores: dict[int, list] = {i: [] for i in range(len(layers))}
    handles = [
        layer.self_attn.o_proj.register_forward_hook(
            lambda _m, _i, out, idx=i: stores[idx].append(out.detach().float())
        )
        for i, layer in enumerate(layers)
    ]

    probes = [json.loads(l) for l in args.probes.open()]
    per_layer_mse = torch.zeros(len(layers))
    try:
        for probe in tqdm(probes, desc="probes"):
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in (probe["self_prompt"], probe["other_prompt"])
            ]
            enc = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            with torch.no_grad():
                model(**enc)
            for i in range(len(layers)):
                a_self, a_other = stores[i].pop()
                per_layer_mse[i] += torch.nn.functional.mse_loss(a_self, a_other).cpu()
    finally:
        for handle in handles:
            handle.remove()

    per_layer_mse /= len(probes)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "n_probes": len(probes),
        "per_layer_mse": [round(v, 6) for v in per_layer_mse.tolist()],
        "mean_mse": round(per_layer_mse.mean().item(), 6),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"mean layer-wise MSE: {result['mean_mse']}  (saved {out_path})")


if __name__ == "__main__":
    main()
