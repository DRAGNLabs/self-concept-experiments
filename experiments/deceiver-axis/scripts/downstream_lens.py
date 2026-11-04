"""Causal downstream lens: add a steering vector at one layer on the real turn-0 task prompts (HF, eager hooks, no
generation) and record what the later layers make of it, arm by arm, against the unsteered forward pass of the same
prompt. Per record layer L and prompt (token mean over the prompt, BOS excluded):

  rel_norm   ||h_L^steered - h_L|| / ||h_L||            how big the downstream difference is
  amp        ||Δ_L|| / |coefficient|                     growth of the perturbation relative to its injected size (1 at the steer layer)
  cos_steer  cos(Δ_L, u_injected)                        how much of Δ still points along the injected direction
  cos_axis   cos(Δ_L, axis_L)                            alignment of Δ with a reference axis (gi_v2) at that layer
  frac_pc    ||P_L Δ_L||² / ||Δ_L||²                     share of Δ inside the top-k PCs of that layer's response means (on-manifold share)
plus, on the last --kl-positions prompt positions, KL(p_unsteered || p_steered) and the top-1 flip rate of the next-token
distribution. Arms: ``name=seed:<int>:<alpha>`` (random_directions seed), ``name=axis:<path>:<alpha>`` (axis artifact,
unit at the steer layer), ``name=noise:<seed>:<alpha>`` (fresh per-token unit noise, ``intervene.noise_transform``).

    python scripts/downstream_lens.py --model google/gemma-4-31B-it --data <hard108_data> --scenario hard108
        --steer-layer 30 --record-layers 30 33 36 40 45 50 55 59 --layer-norms <layer_norms.pt>
        --act-dir <act_response_only> --pcs 50 --axis gi_v2=<persona_contrast.pt>
        --arm rand1729=seed:1729:0.01 --arm rand1729neg=seed:1729:-0.01 --arm rand1730=seed:1730:0.01 ...
        --out <dir>
Writes downstream_lens.json (per arm × layer means and SEs over prompts, per-prompt arrays), downstream_lens.md and
downstream_lens.png/.pdf.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from selfconcept.codebench import harness
from selfconcept.common.chat import chat_template_kwargs
from selfconcept.common.loading import load_causal_lm
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.capture import capture
from selfconcept.measurement.intervene import _hooks, add_transform, noise_transform
from selfconcept.measurement.unit_axes import load_unit_axes


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scenario", default="hard108")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--impossible-prompt", default="critical_no_carve")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--feedback", default="neutral")
    ap.add_argument("--steer-layer", type=int, default=30)
    ap.add_argument("--record-layers", nargs="+", type=int, default=[30, 33, 36, 40, 45, 50, 55, 59])
    ap.add_argument("--layer-norms", type=Path, required=True)
    ap.add_argument("--act-dir", type=Path, help="extraction cache of response means for the per-layer PC bases")
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--axis", action="append", default=[], help="name=path reference axis for cos_axis (unit per layer)")
    ap.add_argument("--arm", action="append", required=True, help="name=kind:param:alpha, kind in seed/axis/noise")
    ap.add_argument("--noise-bank", type=int, default=8192)
    ap.add_argument("--kl-positions", type=int, default=32)
    ap.add_argument("--max-prompt-tokens", type=int, default=6000)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def model_dims(model_name: str) -> tuple[int, int]:
    config = AutoConfig.from_pretrained(model_name)
    text = getattr(config, "text_config", None) or config
    return int(text.num_hidden_layers), int(text.hidden_size)


def pc_bases(act_dir: Path, layers: list[int], k: int) -> dict[int, np.ndarray]:
    rows = {layer: [] for layer in layers}
    for path in sorted(act_dir.glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        for value in data.values():
            if hasattr(value, "shape") and value.ndim == 2:
                for layer in layers:
                    rows[layer].append(value[layer].float().numpy())
    bases = {}
    for layer in layers:
        matrix = np.stack(rows[layer])
        _, _, vt = np.linalg.svd(matrix - matrix.mean(axis=0), full_matrices=False)
        bases[layer] = vt[:k].astype(np.float32)
    return bases


def main() -> None:
    args = parse_args()
    num_layers, hidden = model_dims(args.model)
    norms = torch.load(args.layer_norms, map_location="cpu", weights_only=False)
    layer_norms = (norms["layer_norms"] if isinstance(norms, dict) else norms).float()
    layer_norm = float(layer_norms[args.steer_layer])
    device = torch.device("cuda:0")

    axes = {}
    for spec in args.axis:
        name, path = spec.split("=", 1)
        units = load_unit_axes(Path(path), args.record_layers, num_layers, hidden, args.model)
        axes[name] = {layer: torch.as_tensor(units[layer]["direction"]).to(device) for layer in args.record_layers}

    arms = []
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        kind, param, alpha = rest.split(":")
        alpha = float(alpha)
        coefficient = alpha * layer_norm
        if kind == "seed":
            unit = torch.as_tensor(random_directions(hidden, 1, int(param), args.steer_layer)[0]).float().to(device)
            transform = add_transform(unit, coefficient)
        elif kind == "axis":
            units = load_unit_axes(Path(param), [args.steer_layer], num_layers, hidden, args.model)
            unit = torch.as_tensor(units[args.steer_layer]["direction"]).float().to(device)
            transform = add_transform(unit, coefficient)
        elif kind == "noise":
            bank = torch.as_tensor(random_directions(hidden, args.noise_bank, int(param), args.steer_layer)).float().to(device)
            unit = None
            transform = noise_transform(bank, coefficient)
        else:
            raise SystemExit(f"unknown arm kind {kind} in {spec}")
        arms.append({"name": name, "kind": kind, "param": param, "alpha": alpha, "coefficient": coefficient,
                     "unit": unit, "transform": transform})

    bases = {}
    if args.act_dir is not None:
        print("computing PC bases", file=sys.stderr, flush=True)
        bases = {layer: torch.as_tensor(basis).to(device) for layer, basis in pc_bases(args.act_dir, args.record_layers, args.pcs).items()}

    print("loading model", file=sys.stderr, flush=True)
    model = load_causal_lm(args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    instruction = harness.instruction_by_impossible_prompt[args.impossible_prompt]
    examples = harness.load_examples(args.scenario, args.data, args.n)

    metrics = ["rel_norm", "amp", "cos_steer", "frac_pc"] + [f"cos_{name}" for name in axes]
    per_prompt = {arm["name"]: {layer: {m: [] for m in metrics} for layer in args.record_layers} for arm in arms}
    kl_rows = {arm["name"]: {"kl": [], "flip": [], "kl_last": [], "flip_last": []} for arm in arms}
    example_ids, prompt_tokens = [], []
    for index, example in enumerate(examples):
        message = harness.check_task_message(example, args.max_attempts, instruction, args.feedback)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True,
                                            **chat_template_kwargs(), tokenize=True, return_dict=False)
        if len(ids) > args.max_prompt_tokens:
            print(f"{example['example_id']}: {len(ids)} tokens, skipped", file=sys.stderr)
            continue
        example_ids.append(example["example_id"])
        prompt_tokens.append(len(ids))
        ids = torch.tensor([ids], device=device)
        with torch.inference_mode(), capture(model, args.record_layers) as base_capture:
            base_logits = model(ids).logits[0, -args.kl_positions:].float()
        base = {layer: base_capture[layer][0].float() for layer in args.record_layers}
        base_logp = torch.log_softmax(base_logits, dim=-1)
        base_top = base_logits.argmax(dim=-1)
        for arm in arms:
            with torch.inference_mode(), _hooks(model, {args.steer_layer: arm["transform"]}), capture(model, args.record_layers) as arm_capture:
                arm_logits = model(ids).logits[0, -args.kl_positions:].float()
            arm_logp = torch.log_softmax(arm_logits, dim=-1)
            kl = (base_logp.exp() * (base_logp - arm_logp)).sum(dim=-1)
            flips = (arm_logits.argmax(dim=-1) != base_top).float()
            kl_rows[arm["name"]]["kl"].append(float(kl.mean()))
            kl_rows[arm["name"]]["flip"].append(float(flips.mean()))
            kl_rows[arm["name"]]["kl_last"].append(float(kl[-1]))
            kl_rows[arm["name"]]["flip_last"].append(float(flips[-1]))
            for layer in args.record_layers:
                delta = (arm_capture[layer][0].float() - base[layer])[1:]  # BOS excluded
                h = base[layer][1:]
                dn = delta.norm(dim=-1).clamp_min(1e-9)
                row = per_prompt[arm["name"]][layer]
                row["rel_norm"].append(float((dn / h.norm(dim=-1).clamp_min(1e-9)).mean()))
                row["amp"].append(float(dn.mean() / abs(arm["coefficient"])))
                row["cos_steer"].append(float(((delta @ arm["unit"]) / dn).mean()) if arm["unit"] is not None else float("nan"))
                if layer in bases:
                    proj = delta @ bases[layer].T
                    row["frac_pc"].append(float(((proj ** 2).sum(dim=-1) / (dn ** 2)).mean()))
                else:
                    row["frac_pc"].append(float("nan"))
                for name, units in axes.items():
                    row[f"cos_{name}"].append(float(((delta @ units[layer]) / dn).mean()))
        if index % 10 == 0:
            print(f"{index + 1}/{len(examples)} prompts", file=sys.stderr, flush=True)

    def summarize(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        return {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"), "n": int(len(arr))} if len(arr) else {"mean": float("nan"), "se": float("nan"), "n": 0}

    summary = {}
    for arm in arms:
        entry = {"kind": arm["kind"], "param": arm["param"], "alpha": arm["alpha"], "coefficient": arm["coefficient"],
                 "layers": {str(layer): {m: summarize(per_prompt[arm["name"]][layer][m]) for m in metrics} for layer in args.record_layers},
                 "output": {k: summarize(v) for k, v in kl_rows[arm["name"]].items()}}
        summary[arm["name"]] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "downstream_lens.json").write_text(json.dumps({
        "model": args.model, "scenario": args.scenario, "steer_layer": args.steer_layer, "layer_norm": layer_norm,
        "record_layers": args.record_layers, "pcs": args.pcs, "kl_positions": args.kl_positions, "examples": example_ids,
        "prompt_tokens": prompt_tokens, "summary": summary,
        "per_prompt": {arm: {str(layer): rows for layer, rows in layers.items()} for arm, layers in per_prompt.items()},
        "per_prompt_output": kl_rows}, indent=1))
    write_md(args, summary, metrics, len(example_ids))
    chart(args, summary, metrics)
    print(json.dumps({arm: {"kl": round(e["output"]["kl"]["mean"], 4), "flip": round(e["output"]["flip"]["mean"], 4),
                            "amp_last": round(e["layers"][str(args.record_layers[-1])]["amp"]["mean"], 2)} for arm, e in summary.items()}, indent=1))


def write_md(args, summary, metrics, n_prompts) -> None:
    lines = [f"# Causal downstream lens: steer at L{args.steer_layer} on {n_prompts} {args.scenario} turn-0 prompts ({args.model})", "",
             f"Token means over each prompt (BOS excluded), then mean ± SE over prompts. amp = ||Δ|| / |coefficient| (1 at the steer layer "
             f"by construction). frac_pc = share of Δ inside the top-{args.pcs} PCs of that layer's response means. KL / flip on the last "
             f"{args.kl_positions} prompt positions.", ""]
    lines += ["| arm | kind | α | coefficient | KL mean | KL last | top-1 flip | flip last |", "|---|---|---|---|---|---|---|---|"]
    for arm, e in summary.items():
        o = e["output"]
        lines.append(f"| {arm} | {e['kind']} {e['param']} | {e['alpha']:+g} | {e['coefficient']:+.2f} | {o['kl']['mean']:.4f} ± {o['kl']['se']:.4f} | "
                     f"{o['kl_last']['mean']:.4f} | {o['flip']['mean']:.3f} | {o['flip_last']['mean']:.3f} |")
    for metric in metrics:
        lines += ["", f"## {metric}", "", "| arm | " + " | ".join(f"L{l}" for l in args.record_layers) + " |", "|---|" + "---|" * len(args.record_layers)]
        for arm, e in summary.items():
            cells = []
            for layer in args.record_layers:
                s = e["layers"][str(layer)][metric]
                cells.append("n/a" if np.isnan(s["mean"]) else (f"{s['mean']:.3f}" if abs(s["mean"]) < 100 else f"{s['mean']:.0f}"))
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    (args.out / "downstream_lens.md").write_text("\n".join(lines) + "\n")


def chart(args, summary, metrics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, RULE = "#fcfcfb", "#1c1b19", "#52514e", "#dedcd7"
    palette = ["#1f4e9e", "#b8651b", "#2f7d4f", "#8a3a8a", "#c23b3b", "#6b6b6b", "#0f8b8d", "#9a7b0a", "#5c4a9a"]
    layers = args.record_layers
    fig = plt.figure(figsize=(14, 9.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.6], hspace=0.45, wspace=0.3, left=0.06, right=0.985, top=0.9, bottom=0.04)
    panels = [("amp", "growth of Δ: ||Δ|| / |coefficient|", True), ("frac_pc", f"share of Δ in the top-{args.pcs} PCs", False),
              ("cos_steer", "cos(Δ, injected direction)", False)]
    arm_names = list(summary)
    for col, (metric, title, log) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.set_facecolor(SURFACE)
        for i, arm in enumerate(arm_names):
            ys = [summary[arm]["layers"][str(l)][metric]["mean"] for l in layers]
            es = [summary[arm]["layers"][str(l)][metric]["se"] for l in layers]
            if all(np.isnan(ys)):
                continue
            ax.errorbar(layers, ys, yerr=es, color=palette[i % len(palette)], lw=2, marker="o", ms=5, capsize=2, label=arm)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("decoder layer", color=INK2)
        ax.set_xticks(layers)
        ax.grid(True, color=RULE, lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK2)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    last = str(layers[-1])
    header = ["arm", "kind", "α", "KL mean", "KL last", "top-1 flip", f"amp L{last}", f"frac_pc L{last}", f"cos_steer L{last}"]
    rows = []
    for arm in arm_names:
        e = summary[arm]
        o = e["output"]
        L = e["layers"][last]
        rows.append([arm, f"{e['kind']} {e['param']}", f"{e['alpha']:+g}", f"{o['kl']['mean']:.4f} ± {o['kl']['se']:.4f}",
                     f"{o['kl_last']['mean']:.4f}", f"{o['flip']['mean']:.3f}", f"{L['amp']['mean']:.1f}",
                     "n/a" if np.isnan(L["frac_pc"]["mean"]) else f"{L['frac_pc']['mean']:.3f}",
                     "n/a" if np.isnan(L["cos_steer"]["mean"]) else f"{L['cos_steer']['mean']:.3f}"])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                     colWidths=[0.14, 0.12, 0.07, 0.15, 0.1, 0.1, 0.1, 0.11, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_facecolor(SURFACE if r else "#f0efeb")
        cell.get_text().set_color(INK)
    fig.suptitle(f"Causal downstream lens: steering at L{args.steer_layer} on the {args.scenario} turn-0 prompts, {args.model}",
                 fontsize=13, color=INK, x=0.06, ha="left")
    fig.savefig(args.out / "downstream_lens.png", dpi=160, facecolor=SURFACE)
    fig.savefig(args.out / "downstream_lens.pdf", facecolor=SURFACE)


if __name__ == "__main__":
    main()
