"""What is in a seeded random steering direction? Logit lens and a named-direction dictionary (CPU).

For each `--random-direction-seed` used as a control arm (the same `random_directions(hidden, 1, seed, layer)` draw that
`steered_rh_vllm.py` adds to the residual stream) this decodes the unit direction through the model's final RMSNorm gain
and unembedding, logits = W_U (g * v), and lists the tokens at both ends (ranked by logit and by cosine, as in
`logit_lens.py`), with the isotropic-random null for how peaked a side can be by chance. It then compares the direction
against a dictionary of named directions at the same layer — any axis artifacts given with `--axis name=path` and every
`<role>.pt` minus `default.pt` in `--vectors-dir` — reporting cosines with their z against the isotropic null
(sd = 1 / sqrt(hidden)), plus the cosines between the random directions themselves. The named axes are decoded the same
way next to the random directions so the reports can be read side by side.

    python experiments/deceiver-axis/scripts/random_direction_lens.py --model meta-models/Muse-Glimmer-30B --layer 26 \
        --seeds 1729 1730 1731 --axis gi_v2=<...>/gi_v2/direction/persona_contrast.pt --vectors-dir <...>/vectors_response_only --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logit_lens import lens_direction, load_unembedding, random_extreme_z  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from selfconcept.assistant_axis.internals.model_specifics.registry import unembedding_specifics_for  # noqa: E402
from selfconcept.correlation.random_vectors import random_directions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_axis(path: Path, layer: int) -> torch.Tensor:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    axis = artifact["axis"] if isinstance(artifact, dict) else artifact
    return axis[layer].float()


def role_dictionary(vectors_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    default = torch.load(vectors_dir / "default.pt", map_location="cpu", weights_only=False)["vector"][layer].float()
    out = {}
    for path in sorted(vectors_dir.glob("*.pt")):
        if path.stem == "default" or "minus_default" in path.stem or path.stem in {"persona_contrast", "persona_raw", "honest_raw"} or path.stem.startswith("persona_contrast"):
            continue
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if not (isinstance(artifact, dict) and "vector" in artifact and artifact["vector"].ndim == 2):
            continue
        out[f"{path.stem} − default"] = artifact["vector"][layer].float() - default
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def side_line(side_tokens, shown: int) -> str:
    return ", ".join(f"`{json.dumps(t['token'], ensure_ascii=False)[1:-1]}`" for t in side_tokens[:shown])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--axis", nargs="*", default=[], help="name=path of build_vectors.py artifacts to decode and compare against")
    ap.add_argument("--vectors-dir", type=Path, default=None, help="per-role mean vectors (<role>.pt with default.pt) for the dictionary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--tokens-shown", type=int, default=40)
    ap.add_argument("--null-directions", type=int, default=200)
    ap.add_argument("--null-seed", type=int, default=0)
    args = ap.parse_args()

    unembedding = load_unembedding(unembedding_specifics_for(args.model), args.model)
    hidden = unembedding["gain_scaled_weight"].shape[1]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    directions: dict[str, torch.Tensor] = {}
    for seed in args.seeds:
        directions[f"random {seed}"] = torch.as_tensor(random_directions(hidden, 1, seed, args.layer)[0]).float()
    axes = {}
    for spec in args.axis:
        name, _, path = spec.partition("=")
        axes[name] = load_axis(Path(path), args.layer)
        directions[name] = axes[name]
    roles = role_dictionary(args.vectors_dir, args.layer) if args.vectors_dir else {}

    lenses = [lens_direction(unembedding, tokenizer, v, args.layer, name, args.top_k) for name, v in directions.items()]
    null_z = random_extreme_z(unembedding, hidden, args.null_directions, args.null_seed)

    null_sd = 1.0 / math.sqrt(hidden)
    dictionary = {name: {ref: cosine(v, u) for ref, u in {**axes, **roles}.items() if ref != name} for name, v in directions.items() if name.startswith("random")}
    pairwise = {f"{a} vs {b}": cosine(directions[a], directions[b]) for a in directions for b in directions if a < b}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "random_direction_lens.json").write_text(json.dumps({
        "model": args.model, "layer": args.layer, "seeds": args.seeds, "axes": {k: v for k, v in (s.partition("=")[::2] for s in args.axis)},
        "vectors_dir": str(args.vectors_dir), "hidden": hidden, "null_cosine_sd": null_sd,
        "lenses": lenses, "null_extreme_z": null_z, "dictionary_cosines": dictionary, "pairwise_cosines": pairwise}, indent=1))

    lines = [f"# Random steering directions through the logit lens — {args.model}, layer {args.layer}", "",
             f"Hidden size {hidden}; isotropic null for a cosine between two unrelated unit directions: sd {null_sd:.4f} (|cos| > {2 * null_sd:.3f} is 2 sd).", "",
             f"Null for how peaked one side of a lens can be ({args.null_directions} isotropic random directions, |z| of the most extreme token on a side):"]
    for ranking, zs in null_z.items():
        t = torch.tensor(zs)
        lines.append(f"- {ranking}: median {t.median().item():.2f}, 95th percentile {t.quantile(0.95).item():.2f}, max {t.max().item():.2f} over {len(zs)} sides")
    lines += ["", "## Extreme-token z per direction (the lens's own peakedness)", "", "| direction | logit +side | logit −side | cosine +side | cosine −side |", "|---|---:|---:|---:|---:|"]
    for lens in lenses:
        tb = lens["tokens_by_ranking"]
        lines.append(f"| {lens['direction']} | {tb['logit']['positive'][0]['z']:+.2f} | {tb['logit']['negative'][0]['z']:+.2f} | {tb['cosine']['positive'][0]['z']:+.2f} | {tb['cosine']['negative'][0]['z']:+.2f} |")
    lines += ["", "## Dictionary: cosine of each random direction with the named directions at this layer (z = cos / null sd)", ""]
    refs = list({**axes, **roles}.keys())
    rand_names = [n for n in directions if n.startswith("random")]
    lines.append("| named direction | " + " | ".join(rand_names) + " |")
    lines.append("|---|" + "---:|" * len(rand_names))
    for ref in refs:
        cells = []
        for n in rand_names:
            c = dictionary[n].get(ref)
            cells.append("—" if c is None else f"{c:+.4f} (z {c / null_sd:+.1f})")
        lines.append(f"| {ref} | " + " | ".join(cells) + " |")
    lines += ["", "Pairwise cosines: " + "; ".join(f"{k} {v:+.4f}" for k, v in pairwise.items()), ""]
    for lens in lenses:
        for ranking in ("logit", "cosine"):
            st = lens["tokens_by_ranking"][ranking]
            lines += [f"## {lens['direction']}, ranked by {ranking}", "",
                      f"**positive side** (top z {st['positive'][0]['z']:+.1f}): " + side_line(st["positive"], args.tokens_shown), "",
                      f"**negative side** (top z {st['negative'][0]['z']:+.1f}): " + side_line(st["negative"], args.tokens_shown), ""]
    (args.out / "random_direction_lens.md").write_text("\n".join(lines))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
