"""Measure a LoRA band's mean deltas for band steering (STEER_ROUND.md).

For one adapter and a set of (layer, module) LoRA modules, record the delta the
adapter adds to each projection's output, d(x) = s B A x, on a prompt set:
its mean at the last prompt token ('last', the position the last-token loss
read) and its mean over every valid prompt token ('all'). Also records how
input-dependent each delta is (root-mean-square distance of per-token deltas
from their mean, relative to the mean norm) and the base projection's output
norm at the same positions, so the offset's size can be read against it.

Usage, from a directory holding the pair file and adapters:
  python extract_band_deltas.py --model M --revision R --adapter adapters/seed0 --layers range:19-23 \
      --modules q_proj,v_proj --pairs data/train_soo_pairs.jsonl --out output/band_deltas.pt
"""

import argparse
import json
from pathlib import Path

import torch

from make_constants import encode
from selfconcept.common.loading import load_causal_lm
from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.lora_subset import LAYER_INDEX, parse_layer_spec, parse_module_spec
from selfconcept.soo.steering import offset_key


class DeltaAccumulator:
    def __init__(self):
        self.last_rows, self.last_base_norms = [], []
        self.sum = None
        self.sum_sq = 0.0
        self.base_sq = 0.0
        self.n_tokens = 0

    def add(self, delta: torch.Tensor, base_out: torch.Tensor, mask: torch.Tensor):
        # delta, base_out: [batch, seq, out] float32 on device; mask: [batch, seq]
        m = mask.to(delta.device).bool()
        pos = torch.arange(m.shape[1], device=m.device)
        last = pos.expand_as(m).masked_fill(~m, -1).max(1).values
        rows = torch.arange(delta.shape[0], device=delta.device)
        self.last_rows.append(delta[rows, last].cpu())
        self.last_base_norms.append(base_out[rows, last].norm(dim=-1).cpu())
        valid = delta[m]
        self.sum = valid.sum(0).cpu() if self.sum is None else self.sum + valid.sum(0).cpu()
        self.sum_sq += float(valid.pow(2).sum())
        self.base_sq += float(base_out[m].pow(2).sum())
        self.n_tokens += int(valid.shape[0])

    def finish(self) -> tuple[dict, dict]:
        last = torch.cat(self.last_rows, 0)
        mean_last = last.mean(0)
        spread_last = (last - mean_last).pow(2).sum(1).mean().sqrt()
        stats_last = {
            "n_prompts": int(last.shape[0]), "mean_norm": float(mean_last.norm()),
            "mean_row_norm": float(last.norm(dim=1).mean()), "rms_spread": float(spread_last),
            "rms_spread_over_mean_norm": float(spread_last / mean_last.norm().clamp_min(1e-12)),
            "base_output_norm": float(torch.cat(self.last_base_norms).mean()),
        }
        mean_all = self.sum / self.n_tokens
        msq = self.sum_sq / self.n_tokens
        spread_all = max(msq - float(mean_all.norm()) ** 2, 0.0) ** 0.5
        stats_all = {
            "n_tokens": self.n_tokens, "mean_norm": float(mean_all.norm()), "rms_row_norm": msq ** 0.5,
            "rms_spread": spread_all, "rms_spread_over_mean_norm": spread_all / max(float(mean_all.norm()), 1e-12),
            "base_output_rms_norm": (self.base_sq / self.n_tokens) ** 0.5,
        }
        return {"last": mean_last, "all": mean_all}, {"last": stats_last, "all": stats_all}


def band_lora_modules(peft, keep_layers: set[int], keep_modules: tuple[str, ...]) -> dict[str, torch.nn.Module]:
    from peft.tuners.lora.layer import LoraLayer

    found = {}
    for name, module in peft.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        m = LAYER_INDEX.search(name)
        leaf = name.rsplit(".", 1)[-1]
        if m and int(m.group(1)) in keep_layers and leaf in keep_modules:
            found[offset_key(int(m.group(1)), leaf)] = module
    if not found:
        raise SystemExit("no LoRA modules in the requested band")
    return found


def measure_band(peft, modules: dict[str, torch.nn.Module], batches) -> tuple[dict, dict]:
    """Mean LoRA deltas ('last' / 'all') and their statistics for each named module over the batches."""
    acc = {key: DeltaAccumulator() for key in modules}
    current: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for key, module in modules.items():
        def hook(_m, inputs, output, key=key, module=module):
            x = inputs[0]
            with torch.no_grad():
                base_out = module.base_layer(x)
            current[key] = ((output - base_out).float(), base_out.float())
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for batch in batches:
                current.clear()
                peft(**batch)
                for key in modules:
                    delta, base_out = current[key]
                    acc[key].add(delta, base_out, batch["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    offsets = {"last": {}, "all": {}}
    stats = {"last": {}, "all": {}}
    for key in sorted(modules, key=lambda k: (int(k.split(":")[0]), k)):
        vecs, st = acc[key].finish()
        for mode in ("last", "all"):
            offsets[mode][key] = vecs[mode]
            stats[mode][key] = st[mode]
    return offsets, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--adapter", required=True, help="PEFT adapter directory")
    ap.add_argument("--layers", required=True, help="layer spec as in selfconcept.soo.lora_subset, e.g. range:19-23")
    ap.add_argument("--modules", default="q_proj,v_proj")
    ap.add_argument("--pairs", type=Path, required=True, help="jsonl with self_prompt/other_prompt; both members are used")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = load_causal_lm(args.model, dtype=dtype, revision=args.revision).to(device).eval()
    peft = PeftModel.from_pretrained(model, args.adapter).eval()
    n_layers = len(get_decoder_layers(peft))
    keep_layers = parse_layer_spec(args.layers, n_layers)
    keep_modules = parse_module_spec(args.modules)
    modules = band_lora_modules(peft, keep_layers, keep_modules)
    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    prompts = [r[k] for r in rows for k in ("self_prompt", "other_prompt")]

    offsets, stats = measure_band(peft, modules, encode(tokenizer, prompts, device))
    meta = {"model": args.model, "revision": args.revision, "adapter": args.adapter, "layer_spec": args.layers,
            "layers": sorted(keep_layers), "modules": list(keep_modules), "keys": sorted(modules, key=lambda k: (int(k.split(":")[0]), k)),
            "pairs": str(args.pairs), "n_prompts": len(prompts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**meta, "offsets": offsets, "stats": stats}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({**meta, "stats": stats}, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
