"""Antonym control directions vs the deceiver axis (gi_v2), all four models: per-direction counts (charts in antonym_controls/).

Koby's isolated-text antonym contrasts (AISI method): per WordNet antonym pair, the mean over 5 sentence pairs of
(activation with the word - activation with its antonym), each sentence read alone as a chat message at its final token,
unit length. 100 per model, each added at the deceiver axis's layer and dose for 1 seed (45 episodes on gpt-oss and Muse,
108 on Gemma, 61 on Qwen). Not judged: hack = proxy (shown tests pass, hidden fail); also no-code and hidden pass, as in
anisotropic_counts.py. References: unsteered and the deceiver axis at the same dose (all seeds), from origin/main's
data/controls/cross_model_models.json. Cosine = cosine similarity of each direction with the deceiver axis at that layer.

    python antonym_direction_counts.py antonym_direction_counts.json
"""
import json
import sys
from pathlib import Path

import torch

from anisotropic_counts import count

S = Path("/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis")
MODELS = {
    "gpt-oss-120b": dict(root="gptoss120b", tag="main", layer=17, dose=0.2,
                         unsteered="gptoss120b/main/unsteered", gi="gptoss120b/gi_v2/pilot/gi_p0.2", axis="gptoss120b/gi_v2/direction/persona_contrast.pt"),
    "Muse-Glimmer-30B": dict(root="muse_glimmer", tag="main", layer=26, dose=0.08,
                             unsteered="muse_glimmer/main/unsteered", gi="muse_glimmer/gi_v2/pilot/gi_p0.08", axis="muse_glimmer/gi_v2/direction/persona_contrast.pt"),
    "Gemma-4-31B": dict(root="gemma4_31b_gi_v2", tag="pilot", layer=30, dose=0.08,
                        unsteered="gemma4_31b_gi_v2/rh/hard108/unsteered", gi="gemma4_31b_gi_v2/rh/hard108/L30_a0.08", axis="gemma4_31b_gi_v2/direction/persona_contrast.pt"),
    "Qwen3.8-27B": dict(root="qwen38_27b_gi_v2", tag="pilot", layer=32, dose=0.16,
                        unsteered="qwen38_27b_gi_v2/rh/qwen_hard61/unsteered", gi="qwen38_27b_gi_v2/rh/qwen_hard61/L32_a0.16", axis="qwen38_27b_gi_v2/direction/persona_contrast.pt"),
}


def unit_at_layer(path, layer):
    v = torch.load(path, map_location="cpu", weights_only=False)["axis"][layer].double()
    return v / v.norm()


def main(out):
    result = {}
    for model, c in MODELS.items():
        gi = unit_at_layer(S / c["axis"], c["layer"])
        refs = {"unsteered": count(S / c["unsteered"], c["tag"]), "gi": count(S / c["gi"], c["tag"])}
        dirs = {}
        for i in range(100):
            name = f"antonym_p{i:03d}"
            r = count(S / c["root"] / "antonym_controls/runs" / name, c["tag"], only_seed0=True)
            r["cos_gi_v2"] = float(unit_at_layer(S / c["root"] / "antonym_controls/dirs_isolated" / f"{name}.pt", c["layer"]) @ gi)
            dirs[name] = r
        result[model] = dict(layer=c["layer"], dose=c["dose"], references=refs, directions=dirs)
        rate = lambda r: 100 * r["proxy"] / r["episodes"]
        rates = [rate(r) for r in dirs.values()]
        print(f"{model}: unsteered {rate(refs['unsteered']):.1f}%, deceiver {rate(refs['gi']):.1f}%, controls {min(rates):.1f}-{max(rates):.1f}%, "
              f"at/above {sum(x >= rate(refs['gi']) for x in rates)}/100, median |cos| {sorted(abs(r['cos_gi_v2']) for r in dirs.values())[50]:.3f}")
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
