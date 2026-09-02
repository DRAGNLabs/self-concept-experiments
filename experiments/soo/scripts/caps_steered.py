"""Capability benchmarks with a steering hook active.

The lm_eval CLI can load PEFT adapters but cannot apply our steering hook,
so this drives lm-eval through its Python API with the hooked model object.
alpha=0 gives the unsteered baseline on the identical code path.

Usage: python scripts/caps_steered.py <model_id> <vectors.pt> <layer> <alpha> <out_json>
           [--tasks arc_challenge,hellaswag,mmlu] [--token-mode last]
"""

import argparse
import json
from pathlib import Path

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from selfconcept.soo.loading import load_causal_lm
from selfconcept.soo.steering import get_vector, load_vectors, steer_o_proj

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_id")
parser.add_argument("vectors", type=Path)
parser.add_argument("layer", type=int)
parser.add_argument("alpha", type=float)
parser.add_argument("out_json", type=Path)
parser.add_argument("--tasks", default="arc_challenge,hellaswag,mmlu")
parser.add_argument("--token-mode", default="last")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = load_causal_lm(args.model_id, dtype=torch.bfloat16).to("cuda")
model.eval()
if args.alpha != 0:
    vector = get_vector(load_vectors(args.vectors), args.layer, args.token_mode)
    steer_o_proj(model, args.layer, vector, args.alpha)
    print(f"Steering active: layer {args.layer}, alpha {args.alpha}, |v|={vector.norm():.3f}")

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","))

out = {
    "model": args.model_id,
    "steering": {"layer": args.layer, "alpha": args.alpha, "token_mode": args.token_mode},
    "results": results["results"],
}
args.out_json.parent.mkdir(parents=True, exist_ok=True)
args.out_json.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(results["results"], indent=2, default=str))
