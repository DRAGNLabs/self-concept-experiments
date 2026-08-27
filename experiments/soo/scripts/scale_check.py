"""Cheap check of Koby's hypothesis: do Mistral and OLMo-2-7B differ in
parameter/activation scale such that lr=1e-4 is effectively 'hotter' on Mistral?

Three CPU-only measurements, no forward passes:
  1. Epoch-loss trajectories (epoch-0 SOO loss ~ activation-difference scale
     at the target layer, since training starts from the base model).
  2. Base-weight scale of the LoRA target matrices (q_proj/v_proj) per layer.
  3. Relative perturbation actually applied by training:
     ||alpha/r * B@A||_F / ||W_base||_F per target matrix, per seed.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from selfconcept.common.paths import experiment_dir

ROOT = experiment_dir("soo")
HUB = Path.home() / ".cache/huggingface/hub"

MODELS = {
    "mistral": "models--mistralai--Mistral-7B-Instruct-v0.2",
    "olmo7b": "models--allenai--OLMo-2-1124-7B-Instruct",
}
RUNS = {
    "mistral": ["mistral-7b", "mistral-7b-lasttok"],
    "olmo7b": ["olmo2-7b"],
}
LAYERS = [0, 10, 19, 30]  # sample a few, incl. the SOO layer (19)


def snapshot_files(repo):
    snaps = sorted(glob.glob(str(HUB / repo / "snapshots" / "*" / "*.safetensors")))
    return snaps


def base_weights(repo, wanted):
    """Load only the requested tensors from a sharded safetensors checkpoint."""
    out = {}
    for f in snapshot_files(repo):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k in wanted:
                    out[k] = sf.get_tensor(k).float().numpy()
    return out


def tstats(t):
    return {"std": float(t.std()), "fro": float(np.linalg.norm(t)), "shape": list(t.shape)}


print("=" * 70)
print("1) Epoch-0 SOO loss (activation-difference scale at layer 19/base model)")
print("=" * 70)
for model, runs in RUNS.items():
    for run in runs:
        for log in sorted((ROOT / "results/checkpoints" / run).glob("seed*/train_log.json")):
            d = json.loads(log.read_text())
            losses = d["epoch_losses"]
            print(f"{run:22s} {log.parent.name}: ep0={losses[0]:.3e}  ep1={losses[1]:.3e}"
                  if len(losses) > 1 else
                  f"{run:22s} {log.parent.name}: ep0={losses[0]:.3e}")

print()
print("=" * 70)
print("2) Base-weight scale of LoRA targets (q_proj / v_proj)")
print("=" * 70)
base = {}
for model, repo in MODELS.items():
    wanted = set()
    for L in LAYERS:
        for proj in ("q_proj", "v_proj"):
            wanted.add(f"model.layers.{L}.self_attn.{proj}.weight")
    base[model] = base_weights(repo, wanted)
    for k in sorted(base[model]):
        s = tstats(base[model][k])
        print(f"{model:8s} {k:45s} std={s['std']:.4f} fro={s['fro']:.1f} shape={s['shape']}")

print()
print("=" * 70)
print("3) Trained LoRA delta relative to base:  ||a/r * B@A|| / ||W||")
print("=" * 70)
for model, runs in RUNS.items():
    for run in runs:
        for seed_dir in sorted((ROOT / "results/checkpoints" / run).glob("seed*")):
            ad = seed_dir / "adapter_model.safetensors"
            if not ad.exists():
                continue
            cfg = json.loads((seed_dir / "adapter_config.json").read_text())
            scale = cfg["lora_alpha"] / cfg["r"]
            with safe_open(str(ad), framework="pt") as sf:
                keys = list(sf.keys())
                rows = []
                for k in keys:
                    if "lora_A" not in k:
                        continue
                    kb = k.replace("lora_A", "lora_B")
                    A = sf.get_tensor(k).float().numpy()
                    B = sf.get_tensor(kb).float().numpy()
                    delta = scale * (B @ A)
                    # base key: strip peft prefixes
                    bk = (k.replace("base_model.model.", "")
                            .replace(".lora_A.weight", ".weight"))
                    layer = int(bk.split("layers.")[1].split(".")[0])
                    if bk in base[model]:
                        rel = np.linalg.norm(delta) / np.linalg.norm(base[model][bk])
                    else:
                        rel = np.nan
                    rows.append((layer, bk, float(np.linalg.norm(delta)), rel))
            rels = [r[3] for r in rows if not np.isnan(r[3])]
            deltas = [r[2] for r in rows]
            l19 = [r for r in rows if r[0] == 19]
            print(f"{run:22s} {seed_dir.name}: mean|delta|={np.mean(deltas):.4f}  "
                  f"mean rel (sampled layers)={np.mean(rels):.2e}  n_mats={len(rows)}")
            for _, bk, dn, rel in l19:
                tag = f"rel={rel:.2e}" if not np.isnan(rel) else ""
                print(f"    L19 {bk.split('self_attn.')[1]:15s} |delta|={dn:.4f} {tag}")
print("done")
