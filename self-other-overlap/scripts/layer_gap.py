"""Measure the base-model self/other activation gap at every decoder layer.

For each layer's self_attn.o_proj output, computes over the SOO training pairs:
  - full MSE over right-padded tensors (what soo_mode=full trains on)
  - last-token MSE (what soo_mode=last_token trains on)
  - activation power (mean square), for a scale-free relative gap

Rationale: Mistral's epoch-0 SOO loss at L19 is ~30x smaller than OLMo's,
suggesting the overlap objective is nearly vacuous there. If some Mistral
layer has an OLMo-sized relative gap, SOO training might actually bite there.

Usage: python scripts/layer_gap.py <model_id> <out_json>
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soo.activations import get_decoder_layers
from soo.loading import load_causal_lm
from soo.train import chat_text, encode_batch

model_id, out_json = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = load_causal_lm(model_id, dtype=dtype).to(device)
model.eval()

pairs = [json.loads(l) for l in Path("data/train_soo_pairs.jsonl").open()]
max_len = max(
    len(tokenizer(chat_text(tokenizer, p), add_special_tokens=False)["input_ids"])
    for pair in pairs
    for p in (pair["self_prompt"], pair["other_prompt"])
)

layers = get_decoder_layers(model)
n_layers = len(layers)
store = {}
handles = []
for li in range(n_layers):
    mod = layers[li].get_submodule("self_attn.o_proj")
    handles.append(mod.register_forward_hook(
        lambda _m, _i, out, li=li: store.__setitem__(li, out.float())
    ))

# accumulators per layer: full-MSE, last-MSE, power (all batch-weighted sums)
acc = {li: [0.0, 0.0, 0.0] for li in range(n_layers)}
n_batches = 0
batch_size = 4
with torch.no_grad():
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        self_enc = encode_batch(tokenizer, [b["self_prompt"] for b in batch], max_len, device)
        other_enc = encode_batch(tokenizer, [b["other_prompt"] for b in batch], max_len, device)
        model(**self_enc)
        a_self = dict(store)
        store.clear()
        model(**other_enc)
        a_other = dict(store)
        store.clear()
        sl = self_enc["attention_mask"].sum(1) - 1
        ol = other_enc["attention_mask"].sum(1) - 1
        ar = torch.arange(len(batch), device=device)
        for li in range(n_layers):
            s, o = a_self[li], a_other[li]
            acc[li][0] += torch.nn.functional.mse_loss(s, o).item()
            acc[li][1] += torch.nn.functional.mse_loss(s[ar, sl], o[ar, ol]).item()
            acc[li][2] += ((s.pow(2).mean() + o.pow(2).mean()) / 2).item()
        n_batches += 1
for h in handles:
    h.remove()

rows = []
for li in range(n_layers):
    full, last, power = (v / n_batches for v in acc[li])
    rows.append({
        "layer": li,
        "full_mse": full,
        "last_mse": last,
        "power": power,
        "rel_full": full / power,
        "rel_last": last / power,
    })
Path(out_json).write_text(json.dumps({"model": model_id, "layers": rows}, indent=2))

print(f"\n{model_id}  ({n_layers} layers, {len(pairs)} pairs)")
print(f"{'L':>3} {'full_mse':>10} {'last_mse':>10} {'power':>10} {'rel_full':>9} {'rel_last':>9}")
for r in rows:
    print(f"{r['layer']:>3} {r['full_mse']:>10.3e} {r['last_mse']:>10.3e} "
          f"{r['power']:>10.3e} {r['rel_full']:>9.4f} {r['rel_last']:>9.4f}")
