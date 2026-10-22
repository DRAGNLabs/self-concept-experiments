"""Capability benchmarks with a steering hook active.

The lm_eval CLI can load PEFT adapters but cannot apply our steering hook,
so this drives lm-eval through its Python API with the hooked model object.
alpha=0 gives the unsteered baseline on the identical code path.

Usage: python scripts/caps_steered.py <model_id> <vectors.pt> <layer> <alpha> <out_json>
           [--tasks arc_challenge,hellaswag,mmlu] [--token-mode last] [--adapter <peft dir>]
           [--positions all|response|prompt]
--positions response/prompt (PositionalSteering) needs the chat template's
generation prompt in every input, so those runs pass apply_chat_template=True
to lm_eval and are only comparable to an alpha=0 run with the same flag.
The lm_eval CLI's peft= path would also load Qwen3.x checkpoints through
AutoModelForCausalLM (silent decoder mismatch), so LoRA capabilities go
through here too, via load_causal_lm + --adapter with alpha=0.
"""

import argparse
import json
from pathlib import Path

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.loading import load_causal_lm
from selfconcept.soo.steering import PositionalSteering, get_vector, load_vectors, response_marker, steer_o_proj

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_id")
parser.add_argument("vectors", type=Path)
parser.add_argument("layer", type=int)
parser.add_argument("alpha", type=float)
parser.add_argument("out_json", type=Path)
parser.add_argument("--tasks", default="arc_challenge,hellaswag,mmlu")
parser.add_argument("--token-mode", default="last")
parser.add_argument("--adapter", type=Path, help="PEFT adapter to load (use alpha=0 for LoRA-only capabilities)")
parser.add_argument("--positions", choices=["all", "response", "prompt"], default="all")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = load_causal_lm(args.model_id, dtype=torch.bfloat16).to("cuda")
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    print(f"Adapter merged: {args.adapter}")
model.eval()
if args.alpha != 0:
    vector = get_vector(load_vectors(args.vectors), args.layer, args.token_mode)
    positions = None
    if args.positions != "all":
        positions = PositionalSteering(response_marker(tokenizer, chat_template_kwargs()), args.positions)
    steer_o_proj(model, args.layer, vector, args.alpha, positions=positions)
    print(f"Steering active: layer {args.layer}, alpha {args.alpha}, |v|={vector.norm():.3f}, positions {args.positions}")

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
eval_kwargs = {"apply_chat_template": True} if args.positions != "all" else {}
results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), **eval_kwargs)

out = {
    "model": args.model_id,
    "steering": {"layer": args.layer, "alpha": args.alpha, "token_mode": args.token_mode, "positions": args.positions},
    "apply_chat_template": bool(eval_kwargs),
    "adapter": str(args.adapter) if args.adapter else None,
    "results": results["results"],
}
args.out_json.parent.mkdir(parents=True, exist_ok=True)
args.out_json.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(results["results"], indent=2, default=str))
