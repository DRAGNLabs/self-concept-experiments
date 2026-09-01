"""Extract self-other steering vectors at every decoder layer.

One forward sweep over the SOO training pairs on the *base* model (no
training) computes, per layer, the mean difference between self- and
other-referencing activations at the self_attn.o_proj output:

    v[layer] = E_pairs[a_self - a_other]

in two token conventions mirroring the two soo_modes trained previously:
  - 'last': difference at each prompt's final non-pad token
  - 'mean': difference of each prompt's mean over its own valid tokens

Output is a single .pt holding vectors for all layers (float32), consumed by
selfconcept.soo.steering / evaluate.py --steer-*.

Usage: python scripts/extract_steering.py <model_id> <out_pt> [--batch-size 4] [--max-pairs N]
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from selfconcept.common.paths import experiment_dir
from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.loading import load_causal_lm
from selfconcept.soo.train import chat_text, encode_batch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_id")
parser.add_argument("out_pt", type=Path)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-pairs", type=int, help="truncate pairs (smoke tests)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = load_causal_lm(args.model_id, dtype=dtype).to(device)
model.eval()

pairs = [json.loads(l) for l in (experiment_dir("soo") / "data/train_soo_pairs.jsonl").open()]
if args.max_pairs:
    pairs = pairs[: args.max_pairs]
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

sum_last = {li: 0.0 for li in range(n_layers)}
sum_mean = {li: 0.0 for li in range(n_layers)}
sum_power = {li: 0.0 for li in range(n_layers)}
n_seen = 0
n_batches = 0
with torch.no_grad():
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i : i + args.batch_size]
        self_enc = encode_batch(tokenizer, [b["self_prompt"] for b in batch], max_len, device)
        other_enc = encode_batch(tokenizer, [b["other_prompt"] for b in batch], max_len, device)
        model(**self_enc)
        a_self = dict(store)
        store.clear()
        model(**other_enc)
        a_other = dict(store)
        store.clear()
        smask = self_enc["attention_mask"]
        omask = other_enc["attention_mask"]
        sl, ol = smask.sum(1) - 1, omask.sum(1) - 1
        ar = torch.arange(len(batch), device=device)
        for li in range(n_layers):
            s, o = a_self[li], a_other[li]
            sum_last[li] += (s[ar, sl] - o[ar, ol]).sum(0).cpu()
            s_mean = (s * smask.unsqueeze(-1)).sum(1) / smask.sum(1, keepdim=True)
            o_mean = (o * omask.unsqueeze(-1)).sum(1) / omask.sum(1, keepdim=True)
            sum_mean[li] += (s_mean - o_mean).sum(0).cpu()
            # per-token mean square at valid positions, for scale context
            valid = smask.sum() + omask.sum()
            sum_power[li] += (
                (s.pow(2).sum(-1) * smask).sum() + (o.pow(2).sum(-1) * omask).sum()
            ).item() / valid.item()
        n_seen += len(batch)
        n_batches += 1
for h in handles:
    h.remove()

vectors_last = torch.stack([sum_last[li] / n_seen for li in range(n_layers)])
vectors_mean = torch.stack([sum_mean[li] / n_seen for li in range(n_layers)])
hidden = vectors_last.shape[1]
# typical activation vector norm ~= sqrt(mean-square-norm), for judging alpha scale
act_norm = torch.tensor([(sum_power[li] / n_batches) ** 0.5 for li in range(n_layers)])

args.out_pt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": args.model_id,
        "n_pairs": n_seen,
        "hidden_size": hidden,
        "vectors_last": vectors_last,
        "vectors_mean": vectors_mean,
        "act_norm": act_norm,
    },
    args.out_pt,
)

print(f"\n{args.model_id}  ({n_layers} layers, {n_seen} pairs, hidden {hidden})")
print(f"{'L':>3} {'|v_last|':>10} {'|v_mean|':>10} {'|act|':>10} {'rel_last':>9} {'rel_mean':>9}")
for li in range(n_layers):
    nl, nm, na = vectors_last[li].norm().item(), vectors_mean[li].norm().item(), act_norm[li].item()
    print(f"{li:>3} {nl:>10.3e} {nm:>10.3e} {na:>10.3e} {nl / na:>9.4f} {nm / na:>9.4f}")
print(f"\nSaved {args.out_pt}")
