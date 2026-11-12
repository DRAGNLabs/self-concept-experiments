"""Hack-direction lens, extraction half: teacher-force finished hard108 episodes through Gemma 4 31B, turn by turn
exactly as the model saw them (history with the earlier finals only, the chat template's own rendering, then this
turn's thought channel and final), and record at the record layers the token means over the reasoning tokens (all
turns pooled, and per turn) and over the answer tokens. For a steered arm every turn goes through twice, once with the
arm's steering hook (as generated) and once without, so the pair gives the steering's downstream perturbation Δ on the
very tokens the model produced; per-token Δ norms, cosines and signed projections onto reference directions (hack
directions, gi_v2, random nulls, the injected vector) are reduced on the GPU. Analysis (hack-minus-no-hack directions,
projections, chart) is hack_lens_analysis.py.

    python scripts/hack_lens_extract.py --model google/gemma-4-31B-it --data <hard108_data> --tag pilot
        --run <arm dir with seedN/> --arm rand1729 --seeds 0 1 --steer seed:1729:0.01 --steer-layer 30
        --layer-norms <layer_norms.pt> --record-layers 30 31 ... 59
        --direction gf_all=<directions_headline.pt>:goodfire_all_response_pl_all_families --direction gi_v2=<persona_contrast.pt>
        --null-directions 8 --out <dir>

--steer none (unsteered arm, one pass per turn), seed:<int>:<alpha> (random_directions at the steer layer),
axis:<path>:<alpha> (unit axis artifact at the steer layer), noise:<seed>:<alpha> (per-token unit noise,
intervene.noise_transform; the generation-time draw is not reproducible, so this is a fresh draw of the same norm).
coefficient = alpha * layer norm. Writes <out>/<arm>__seed<N>.pt per seed (fp16 means, float32 statistics, metadata);
existing files are skipped. Koby's L30 acts (pca/acts) hold one turn per episode; here every turn is read.
"""
import argparse
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from selfconcept.common.loading import load_causal_lm
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.intervene import _hooks, add_transform, noise_transform
from selfconcept.measurement.unit_axes import load_unit_axes
from selfconcept.soo.activations import get_decoder_layers

_spec = importlib.util.spec_from_file_location("make_dossiers", Path(__file__).with_name("make_dossiers.py"))
make_dossiers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_dossiers)

MAX_TURNS = 5
SEGMENTS = ("reasoning", "answer")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tag", default="pilot", help="records file stem: <tag>_impossible_original.jsonl")
    ap.add_argument("--run", type=Path, required=True, help="arm directory holding seed<N>/ subdirectories")
    ap.add_argument("--arm", required=True, help="arm name for the output files")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--steer", default="none", help="none | seed:<int>:<alpha> | axis:<path>:<alpha> | noise:<seed>:<alpha>")
    ap.add_argument("--steer-layer", type=int, default=30)
    ap.add_argument("--layer-norms", type=Path, help="required unless --steer none")
    ap.add_argument("--noise-bank", type=int, default=8192)
    ap.add_argument("--record-layers", nargs="+", type=int, default=list(range(30, 60)))
    ap.add_argument("--direction", action="append", default=[], help="name=path[:key]; (num_layers, hidden) artifact, unit per layer")
    ap.add_argument("--null-directions", type=int, default=8, help="random unit directions per record layer (seed 777)")
    ap.add_argument("--max-length", type=int, default=40960, help="per-turn token cap (the tail is dropped)")
    ap.add_argument("--no-token-delta", action="store_true", help="skip per-token Δ statistics (no per-token base store)")
    ap.add_argument("--n", type=int, default=None, help="episodes per seed (smoke)")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def model_dims(model_name: str) -> tuple[int, int]:
    config = AutoConfig.from_pretrained(model_name)
    text = getattr(config, "text_config", None) or config
    return int(text.num_hidden_layers), int(text.hidden_size)


def load_direction_matrix(spec: str, num_layers: int, hidden: int) -> tuple[str, torch.Tensor]:
    """name=path[:key] -> (name, (num_layers, hidden) float32, rows unit-normalised); key selects an entry of a dict artifact."""
    name, rest = spec.split("=", 1)
    path, _, key = rest.partition(":")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if key:
            data = data[key]
        elif "axis" in data:
            data = data["axis"]
        else:
            raise SystemExit(f"{spec}: dict artifact, give the key (one of {list(data)[:8]})")
    matrix = torch.as_tensor(data).float()
    if matrix.shape != (num_layers, hidden):
        raise SystemExit(f"{spec}: shape {tuple(matrix.shape)}, expected {(num_layers, hidden)}")
    return name, matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def build_steering(args, hidden: int, num_layers: int, device):
    """(transform or None, unit vector or None, metadata)."""
    if args.steer == "none":
        return None, None, {"mode": "none"}
    kind, param, alpha = args.steer.split(":")
    alpha = float(alpha)
    norms = torch.load(args.layer_norms, map_location="cpu", weights_only=False)
    layer_norms = (norms["layer_norms"] if isinstance(norms, dict) else norms).float()
    coefficient = alpha * float(layer_norms[args.steer_layer])
    meta = {"mode": kind, "param": param, "alpha": alpha, "layer": args.steer_layer, "coefficient": coefficient}
    if kind == "seed":
        unit = torch.as_tensor(random_directions(hidden, 1, int(param), args.steer_layer)[0]).float().to(device)
        return add_transform(unit, coefficient), unit, meta
    if kind == "axis":
        units = load_unit_axes(Path(param), [args.steer_layer], num_layers, hidden, args.model)
        unit = torch.as_tensor(units[args.steer_layer]["direction"]).float().to(device)
        return add_transform(unit, coefficient), unit, meta
    if kind == "noise":
        bank = torch.as_tensor(random_directions(hidden, args.noise_bank, int(param), args.steer_layer)).float().to(device)
        meta["noise_bank"] = args.noise_bank
        return noise_transform(bank, coefficient), None, meta
    raise SystemExit(f"unknown steer kind {kind}")


def episode_turns(tokenizer, example: dict, record: dict, reasoning: dict[int, str], max_length: int):
    """One (ids, segment, turn index) per model turn: the chat template's rendering of the history (earlier finals
    only, as the model saw it) plus the generation prompt, then ``<|channel>thought\\n{reasoning}\\n<channel|>{final}``.
    segment: 0 other, 1 reasoning, 2 answer. Also returns the turns whose reasoning is missing from the sidecar."""
    tok = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    history: list[dict] = []
    turns, missing, full_lengths = [], [], []
    for role, text in make_dossiers.conversation(example, record):
        if role == "user":
            history.append({"role": "user", "content": text})
            continue
        turn = int(text)
        final = record["attempts"][turn].get("completion") or ""
        thought = reasoning.get(turn)
        if thought is None:
            missing.append(turn)
            thought = ""
        prompt = list(tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        head, think, mid, fin = tok("<|channel>thought\n"), (tok(thought) if thought else []), tok("\n<channel|>"), tok(final)
        ids = prompt + head + think + mid + fin
        seg = [0] * (len(prompt) + len(head)) + [1] * len(think) + [0] * len(mid) + [2] * len(fin)
        full_lengths.append(len(ids))
        turns.append((ids[:max_length], np.asarray(seg[:max_length], dtype=np.int8), turn))
        history.append({"role": "assistant", "content": final})
    return turns, missing, full_lengths


@contextmanager
def layer_callbacks(model, callbacks: dict[int, object]):
    decoder_layers = get_decoder_layers(model)
    handles = []
    for layer, callback in callbacks.items():
        def hook(_module, _inputs, output, layer=layer, callback=callback):
            hidden = output[0] if isinstance(output, tuple) else output
            callback(layer, hidden[0])
        handles.append(decoder_layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class Accumulator:
    """Token sums per (segment, layer) over the turns of one episode, on the GPU; means at the end."""

    def __init__(self, L: int, hidden: int, K: int, device, paired: bool, steered: bool):
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)  # noqa: E731
        self.n = {s: z(L) for s in SEGMENTS}
        self.sum = {s: z(L, hidden) for s in SEGMENTS}
        self.norm = {s: z(L) for s in SEGMENTS}
        self.proj = {s: z(L, K) for s in SEGMENTS}
        self.turn_n = z(MAX_TURNS, L)
        self.turn_sum = z(MAX_TURNS, L, hidden)
        self.steered = steered
        if steered:
            self.sum_base = {s: z(L, hidden) for s in SEGMENTS}
            self.norm_base = {s: z(L) for s in SEGMENTS}
            self.proj_base = {s: z(L, K) for s in SEGMENTS}
        self.paired = paired
        if paired:
            self.dnorm = {s: z(L) for s in SEGMENTS}
            self.dproj = {s: z(L, K) for s in SEGMENTS}
            self.dcos = {s: z(L, K) for s in SEGMENTS}
            self.dsum = {s: z(L, hidden) for s in SEGMENTS}

    def add_base(self, li, hf, masks, turn, dmat, base_store, layer):
        """Unsteered pass: for an unsteered arm this is the only pass (goes to the main slots); for a steered arm it
        is the comparison pass (base slots) and the per-token store for Δ."""
        if self.paired:
            base_store[layer] = hf.to(torch.bfloat16)
        for s, mask in masks.items():
            if not int(mask.sum()):
                continue
            hm = hf[mask]
            if self.steered:
                self.sum_base[s][li] += hm.sum(0); self.norm_base[s][li] += hm.norm(dim=-1).sum(); self.proj_base[s][li] += (hm @ dmat).sum(0)
            else:
                self._add_main(s, li, hm, dmat, turn, mask)

    def _add_main(self, s, li, hm, dmat, turn, mask):
        self.n[s][li] += hm.shape[0]; self.sum[s][li] += hm.sum(0); self.norm[s][li] += hm.norm(dim=-1).sum(); self.proj[s][li] += (hm @ dmat).sum(0)
        if s == "reasoning" and 0 <= turn < MAX_TURNS:
            self.turn_n[turn, li] += hm.shape[0]; self.turn_sum[turn, li] += hm.sum(0)

    def add_steered(self, li, hf, masks, turn, dmat, base_store, layer):
        for s, mask in masks.items():
            if not int(mask.sum()):
                continue
            hm = hf[mask]
            self._add_main(s, li, hm, dmat, turn, mask)
            if self.paired:
                delta = hm - base_store[layer][mask].float()
                dn = delta.norm(dim=-1).clamp_min(1e-9)
                dp = delta @ dmat
                self.dnorm[s][li] += dn.sum(); self.dproj[s][li] += dp.sum(0); self.dcos[s][li] += (dp / dn[:, None]).sum(0); self.dsum[s][li] += delta.sum(0)

    def finish(self, store, e, dmat_all):
        c = lambda t: t.cpu()  # noqa: E731
        for s in SEGMENTS:
            n = self.n[s].clamp_min(1)[:, None]
            store[f"{s}_mean"][e] = c(self.sum[s] / n).to(torch.float16)
            st = store["stats"][s]
            st["n"][e] = c(self.n[s]).numpy()
            st["h_norm"][e] = c(self.norm[s] / n[:, 0]).numpy(); st["proj"][e] = c(self.proj[s] / n).numpy()
            if self.steered:
                store[f"{s}_mean_base"][e] = c(self.sum_base[s] / n).to(torch.float16)
                st["h_norm_base"][e] = c(self.norm_base[s] / n[:, 0]).numpy(); st["proj_base"][e] = c(self.proj_base[s] / n).numpy()
            if self.paired:
                st["delta_norm"][e] = c(self.dnorm[s] / n[:, 0]).numpy(); st["delta_proj"][e] = c(self.dproj[s] / n).numpy()
                st["delta_cos"][e] = c(self.dcos[s] / n).numpy()
                md = self.dsum[s] / n
                store[f"delta_{s}_mean"][e] = c(md)
                st["cos_mean_delta"][e] = c(torch.einsum("lh,lhk->lk", md, dmat_all) / md.norm(dim=-1, keepdim=True).clamp_min(1e-9)).numpy()
        tn = self.turn_n.clamp_min(1)[:, :, None]
        store["turn_reasoning_mean"][e] = c(self.turn_sum / tn).to(torch.float16)
        store["turn_n_tokens"][e] = c(self.turn_n).numpy()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    num_layers, hidden = model_dims(args.model)
    device = torch.device("cuda:0")
    layers = args.record_layers
    L = len(layers)
    transform, injected, steer_meta = build_steering(args, hidden, num_layers, device)
    steered = transform is not None
    paired = steered and not args.no_token_delta

    directions: dict[str, torch.Tensor] = {}
    for spec in args.direction:
        name, matrix = load_direction_matrix(spec, num_layers, hidden)
        directions[name] = matrix
    for k in range(args.null_directions):
        directions[f"null_{k}"] = torch.stack([torch.as_tensor(random_directions(hidden, args.null_directions, 777, layer)[k]) for layer in range(num_layers)])
    if injected is not None:
        directions["injected"] = injected.cpu().unsqueeze(0).repeat(num_layers, 1)
    names = list(directions)
    K = len(names)
    dmat = {layer: torch.stack([directions[n][layer] for n in names], dim=1).to(device) for layer in layers}  # (hidden, K)
    dmat_all = torch.stack([dmat[layer] for layer in layers])  # (L, hidden, K)

    print("loading model", file=sys.stderr, flush=True)
    model = load_causal_lm(args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for seed in args.seeds:
        out_path = args.out / f"{args.arm}__seed{seed}.pt"
        if out_path.exists():
            print(f"{out_path.name}: exists, skipped", file=sys.stderr, flush=True)
            continue
        run_dir = args.run / f"seed{seed}"
        records = [json.loads(line) for line in (run_dir / f"{args.tag}_impossible_original.jsonl").open()]
        if args.n:
            records = records[:args.n]
        sidecar: dict[str, dict[int, str]] = {}
        for path in run_dir.glob(f"{args.tag}_*_reasoning.jsonl"):
            for line in path.open():
                entry = json.loads(line)
                sidecar.setdefault(entry["example_id"], {})[int(entry["turn"])] = entry["reasoning"]
        E = len(records)
        store = {
            "keys": [], "example_id": [], "verdict_key": [], "passed": [], "passed_original": [], "n_attempts": [],
            "turn_lengths": [], "truncated_turns": [], "missing_reasoning": [],
            "reasoning_mean": torch.zeros(E, L, hidden, dtype=torch.float16), "answer_mean": torch.zeros(E, L, hidden, dtype=torch.float16),
            "turn_reasoning_mean": torch.zeros(E, MAX_TURNS, L, hidden, dtype=torch.float16), "turn_n_tokens": np.zeros((E, MAX_TURNS, L), np.float32),
            "stats": {s: {"n": np.zeros((E, L), np.float32), "h_norm": np.full((E, L), np.nan, np.float32), "proj": np.full((E, L, K), np.nan, np.float32)} for s in SEGMENTS},
        }
        if steered:
            store["reasoning_mean_base"] = torch.zeros(E, L, hidden, dtype=torch.float16)
            store["answer_mean_base"] = torch.zeros(E, L, hidden, dtype=torch.float16)
            for s in SEGMENTS:
                store["stats"][s].update({"proj_base": np.full((E, L, K), np.nan, np.float32), "h_norm_base": np.full((E, L), np.nan, np.float32)})
                if paired:
                    store["stats"][s].update({k: np.full((E, L), np.nan, np.float32) for k in ("delta_norm",)})
                    store["stats"][s].update({k: np.full((E, L, K), np.nan, np.float32) for k in ("delta_cos", "delta_proj", "cos_mean_delta")})
                    store[f"delta_{s}_mean"] = torch.zeros(E, L, hidden, dtype=torch.float32)  # float32: a Δ of two fp16 means would be rounding noise
        t0 = time.time()
        for e, record in enumerate(records):
            example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
            turns, missing, full_lengths = episode_turns(tokenizer, example, record, sidecar.get(record["example_id"], {}), args.max_length)
            store["keys"].append(f"{args.arm}/seed{seed}/{record['example_id']}")
            store["example_id"].append(record["example_id"])
            store["verdict_key"].append(f"{args.run.name}/seed{seed}/{record['example_id']}")
            store["passed"].append(bool(record.get("passed"))); store["passed_original"].append(bool(record.get("passed_original")))
            store["n_attempts"].append(int(record.get("n_attempts") or len(record["attempts"])))
            store["turn_lengths"].append(full_lengths); store["truncated_turns"].append([t for t, n in enumerate(full_lengths) if n > args.max_length])
            store["missing_reasoning"].append(missing)
            acc = Accumulator(L, hidden, K, device, paired, steered)
            for ids, seg, turn in turns:
                ids_t = torch.tensor([ids], device=device)
                seg_t = torch.as_tensor(seg, device=device)
                masks = {"reasoning": seg_t == 1, "answer": seg_t == 2}
                base_store: dict[int, torch.Tensor] = {}
                with torch.inference_mode():
                    with layer_callbacks(model, {layer: (lambda layer, h, turn=turn, masks=masks, base_store=base_store:
                                                         acc.add_base(layers.index(layer), h.detach().float(), masks, turn, dmat[layer], base_store, layer)) for layer in layers}):
                        model(input_ids=ids_t, logits_to_keep=1)
                    if steered:
                        with _hooks(model, {args.steer_layer: transform}), layer_callbacks(model, {layer: (lambda layer, h, turn=turn, masks=masks, base_store=base_store:
                                                         acc.add_steered(layers.index(layer), h.detach().float(), masks, turn, dmat[layer], base_store, layer)) for layer in layers}):
                            model(input_ids=ids_t, logits_to_keep=1)
                base_store.clear()
            acc.finish(store, e, dmat_all)
            del acc
            if e % 10 == 0 or e == E - 1:
                print(f"{args.arm} seed {seed}: {e + 1}/{E} episodes ({len(turns)} turns, {sum(full_lengths)} tokens), {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        store.update({"arm": args.arm, "seed": seed, "run": str(args.run), "tag": args.tag, "model": args.model, "steer": steer_meta,
                      "record_layers": layers, "directions": names, "max_length": args.max_length, "paired": paired, "seconds": time.time() - t0})
        torch.save(store, out_path)
        print(f"wrote {out_path} ({E} episodes, {store['seconds']:.0f} s, truncated turns {sum(len(t) for t in store['truncated_turns'])}, "
              f"missing reasoning turns {sum(len(m) for m in store['missing_reasoning'])})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
