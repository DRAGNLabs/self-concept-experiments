"""Measure the constants for a no-training replacement round (CONSTANT_ROUND.md).

For one model and layer, record the mean attention-output vector at the last
prompt token over a prompt set, for the base model and for each given LoRA
adapter. The adapter means are the constants the last-token LoRA loss converged
to (FINDINGS2.md rounds 2-2c); the base mean is a content-matched but
prompt-independent alternative. Also records how constant each source actually
is on these prompts (root-mean-square distance of per-prompt outputs from their
mean, relative to the mean norm).

Usage, from a directory holding the pair file and adapters:
  python make_constants.py --model M --revision R --layer L --pairs data/train_soo_pairs.jsonl \
      --adapter adapter_seed0=adapters/seed0 [--adapter adapter_seed1=...] --out constants.pt
"""

import argparse
import json
from pathlib import Path

import torch

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.loading import load_causal_lm
from selfconcept.soo.activations import attn_out_proj, get_decoder_layers


def last_token_outputs(model, layer: int, batches) -> torch.Tensor:
    """Attention-output vectors at each prompt's last valid token; [n_prompts, hidden], float32 on CPU."""
    module = attn_out_proj(get_decoder_layers(model)[layer])
    captured = []
    handle = module.register_forward_hook(lambda _m, _i, output: captured.append(output))
    rows = []
    try:
        with torch.no_grad():
            for batch in batches:
                captured.clear()
                model(**batch)
                out = captured[-1]
                mask = batch["attention_mask"]
                pos = torch.arange(mask.shape[1], device=mask.device)
                last = pos.expand_as(mask).masked_fill(~mask.bool(), -1).max(1).values
                rows.append(out[torch.arange(out.shape[0], device=out.device), last.to(out.device)].float().cpu())
    finally:
        handle.remove()
    return torch.cat(rows, 0)


def collect(model, layer: int, batches) -> torch.Tensor:
    return last_token_outputs(model, layer, batches)


def describe(vectors: torch.Tensor) -> dict:
    mean = vectors.mean(0)
    spread = (vectors - mean).pow(2).sum(1).mean().sqrt()
    return {
        "n_prompts": int(vectors.shape[0]),
        "mean_norm": float(mean.norm()),
        "mean_row_norm": float(vectors.norm(dim=1).mean()),
        "rms_spread": float(spread),
        "rms_spread_over_mean_norm": float(spread / mean.norm().clamp_min(1e-12)),
    }


def encode(tokenizer, prompts, device, batch_size=8):
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, **chat_template_kwargs()) for p in prompts]
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        yield {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="jsonl with self_prompt/other_prompt; both members are used")
    ap.add_argument("--adapter", action="append", default=[],
                    help="NAME=PATH[@LAYERS[@MODULES]] of a PEFT adapter; repeatable. LAYERS/MODULES restrict the "
                         "adapter as in selfconcept.soo.lora_subset (e.g. adapters/seed0@except:32, adapters/seed0@all@q_proj)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = load_causal_lm(args.model, dtype=dtype, revision=args.revision).to(device).eval()
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    prompts = [r[k] for r in rows for k in ("self_prompt", "other_prompt")]
    batches = lambda: encode(tokenizer, prompts, device)  # noqa: E731

    constants, stats = {}, {}
    base = collect(model, args.layer, batches())
    constants["base_mean"] = base.mean(0)
    stats["base_mean"] = describe(base)
    for spec in args.adapter:
        name, target = spec.split("=", 1)
        path, *subset = target.split("@")
        from peft import PeftModel

        from selfconcept.soo.lora_subset import apply_adapter_subset

        peft = PeftModel.from_pretrained(model, path)
        subset_stats = apply_adapter_subset(peft, subset[0] if subset else None, subset[1] if len(subset) > 1 else None)
        peft.eval()
        vecs = collect(peft, args.layer, batches())
        constants[name] = vecs.mean(0)
        stats[name] = describe(vecs)
        if subset_stats:
            stats[name]["adapter_subset"] = subset_stats
        stats[name]["distance_to_base_mean"] = float((constants[name] - constants["base_mean"]).norm())
        model = peft.unload()
        model.eval()
    stats["zero"] = {"n_prompts": len(prompts), "mean_norm": 0.0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "revision": args.revision, "layer": args.layer, "pairs": str(args.pairs),
                "n_prompts": len(prompts), "constants": constants, "stats": stats}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"model": args.model, "layer": args.layer, "n_prompts": len(prompts), "stats": stats}, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
