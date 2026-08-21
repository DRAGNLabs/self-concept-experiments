"""Latent SOO: layer-wise MSE between self- and other-referencing activations
on the 52 probe pairs (paper section 3.1.1 / A.1.3, Table 4).

Usage:
    python -m soo.latent_soo --model allenai/OLMo-2-0425-1B-Instruct \
        [--adapter results/checkpoints/olmo2-1b/seed0] [--tag baseline_1b]

Captures the attention (o_proj) and MLP outputs of every decoder layer for
each pair (prompts right-padded to a common length per pair) and reports
per-layer mean MSE plus overall means, saved to results/latent/. The 26
train-vocab and 26 test-vocab probes are reported separately (and combined) —
the training-time SOO loss only ever saw the train vocab. Note our probe set
and capture sites are not identical to the paper's, so values are comparable
across our own runs, not to the paper's Table 4.
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
    sites = {"attn": "self_attn.o_proj", "mlp": "mlp"}
    stores: dict[tuple[str, int], list] = {(s, i): [] for s in sites for i in range(len(layers))}
    handles = [
        layer.get_submodule(path).register_forward_hook(
            lambda _m, _i, out, key=(site, i): stores[key].append(out.detach().float())
        )
        for i, layer in enumerate(layers)
        for site, path in sites.items()
    ]

    probes = [json.loads(l) for l in args.probes.open()]
    splits = sorted({p.get("vocab_split", "all") for p in probes})
    sums = {
        (site, split): torch.zeros(len(layers)) for site in sites for split in splits
    }
    counts = {split: 0 for split in splits}
    try:
        for probe in tqdm(probes, desc="probes"):
            split = probe.get("vocab_split", "all")
            counts[split] += 1
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
            for site in sites:
                for i in range(len(layers)):
                    a_self, a_other = stores[site, i].pop()
                    sums[site, split][i] += torch.nn.functional.mse_loss(a_self, a_other).cpu()
    finally:
        for handle in handles:
            handle.remove()

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "n_probes": counts,
        "sites": {},
    }
    for site in sites:
        combined = sum(sums[site, split] for split in splits) / len(probes)
        result["sites"][site] = {
            "combined_mean_mse": round(combined.mean().item(), 6),
            "per_layer_mse": [round(v, 6) for v in combined.tolist()],
        }
        for split in splits:
            per_layer = sums[site, split] / counts[split]
            result["sites"][site][f"{split}_mean_mse"] = round(per_layer.mean().item(), 6)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    for site in sites:
        print(f"{site}: " + json.dumps({k: v for k, v in result['sites'][site].items() if k != 'per_layer_mse'}))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
