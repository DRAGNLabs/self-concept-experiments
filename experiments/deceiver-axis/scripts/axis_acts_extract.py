"""Unsteered-episode activations for comparing two directions on the residual stream (gi_v2 against the Goodfire
reward-hacking direction), for any of the four model families: teacher-force finished reward-hacking episodes turn by
turn exactly as the model saw them (history with the earlier finals only, the chat template's own rendering with the
run's chat kwargs from $SOO_CHAT_KWARGS, then this turn's reasoning channel and final in the family's raw completion
markup) and record at every layer, per episode and per segment (reasoning tokens, answer tokens, all turns pooled):

- the token mean of the residual stream (fp16; float32 if a coordinate would overflow), so any direction can be
  projected offline;
- for the named directions (--direction) and --null-directions random unit directions: the token mean of the
  projection, the token sum of its square, and the token sums of the products named x every direction. Token-level
  correlations between two named directions, and the random-direction null for each, follow from these sums.

    SOO_CHAT_KWARGS='{"reasoning_effort":"low"}' python scripts/axis_acts_extract.py --model Qwen/Qwen3.8-27B
        --data <qwen_hard61_data> --tag pilot --run <unsteered arm dir with seedN/> --arm unsteered --seeds 8 9
        --direction gi_v2=<persona_contrast.pt> --direction gf_all=<directions_headline.pt>:goodfire_all_response_pl_all_families
        --null-directions 32 --max-length 73728 --out <dir>

Generalises hack_lens_extract.py (Gemma only, record layers 30-59, steered pairs) to the unsteered arm of every
family; the output keeps that script's keys (reasoning_mean, answer_mean, record_layers, example_id, stats) so
hack_lens caches and these read the same way. --dry-run renders and counts every turn without loading the model.
Writes <out>/<arm>__seed<N>.pt per seed; existing files are skipped. A turn that runs out of GPU memory is retried on
its first 80% of tokens (recorded in oom_truncated). Analysis: axis_acts_analysis.py.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from selfconcept.common.chat import chat_template_kwargs
from selfconcept.correlation.random_vectors import random_directions
from selfconcept.measurement.steering_specifics import steering_model_specifics

_spec = importlib.util.spec_from_file_location("hack_lens_extract", Path(__file__).with_name("hack_lens_extract.py"))
hack_lens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hack_lens)
make_dossiers = hack_lens.make_dossiers

SEGMENTS = ("reasoning", "answer")
NULL_SEED = 777
# family -> (reasoning opener, reasoning closer + final opener, end of turn): the raw text the model writes after the
# generation prompt (join_harmony_completion, join_gemma4_completion, join_muse_glimmer_completion; Qwen's generation
# prompt already ends in "<think>\n" when thinking is on, and its template closes the thought as "\n</think>\n\n").
COMPLETION_MARKUP: dict[str, tuple[str, str, str]] = {
    "gpt-oss": ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>", "<|return|>"),
    "gemma4": ("<|channel>thought\n", "\n<channel|>", ""),
    "muse-glimmer": (" to=self<|message|>", "<|eom|><|start|>assistant to=user<|message|>", "<|eot|>"),
    "qwen3.5": ("", "\n</think>\n\n", "<|im_end|>"),
}
# The opener of a turn without reasoning (the model went straight to its answer). Qwen cannot: the prompt opened the thought.
DIRECT_ANSWER_OPENER = {"gpt-oss": "<|channel|>final<|message|>", "gemma4": "", "muse-glimmer": " to=user<|message|>"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--attn-implementation", default=None, help="default: the family's steering_model_specifics kernel")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tag", default="pilot", help="records file stem: <tag>_impossible_original.jsonl")
    ap.add_argument("--run", type=Path, required=True, help="arm directory holding seed<N>/ subdirectories")
    ap.add_argument("--arm", default="unsteered", help="arm name for the output files")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--record-layers", nargs="+", type=int, default=None, help="default: every layer")
    ap.add_argument("--direction", action="append", default=[], help="name=path[:key]; (num_layers, hidden) artifact, unit-normalised per layer here")
    ap.add_argument("--null-directions", type=int, default=32, help=f"random unit directions per layer (seed {NULL_SEED})")
    ap.add_argument("--max-length", type=int, default=40960, help="per-turn token cap (the tail is dropped)")
    ap.add_argument("--n", type=int, default=None, help="episodes per seed (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="render and count every turn, no model")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def render_turn(tokenizer, family: str, history: list[dict], thought: str, final: str, chat_kwargs: dict) -> tuple[list[int], np.ndarray]:
    """(token ids, segment per token) of one model turn; segment 0 other, 1 reasoning, 2 answer."""
    tok = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"] if text else []  # noqa: E731
    opener, closer, end = COMPLETION_MARKUP[family]
    prompt = list(tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_dict=False, **chat_kwargs))
    if not thought and family in DIRECT_ANSWER_OPENER:
        pieces = [(prompt + tok(DIRECT_ANSWER_OPENER[family]), 0), (tok(final), 2), (tok(end), 0)]
    elif not final:  # the reasoning never closed (stopped or cut off mid-thought)
        pieces = [(prompt + tok(opener), 0), (tok(thought), 1)]
    else:
        pieces = [(prompt + tok(opener), 0), (tok(thought), 1), (tok(closer), 0), (tok(final), 2), (tok(end), 0)]
    ids = [token for piece, _ in pieces for token in piece]
    segments = np.concatenate([np.full(len(piece), segment, dtype=np.int8) for piece, segment in pieces])
    return ids, segments


def episode_turns(tokenizer, family: str, example: dict, record: dict, reasoning: dict[int, str], max_length: int, chat_kwargs: dict):
    """One (ids, segments, turn index) per model turn, plus the turns whose reasoning is missing from the sidecar and
    the untruncated lengths."""
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
        ids, segments = render_turn(tokenizer, family, history, thought, final, chat_kwargs)
        full_lengths.append(len(ids))
        turns.append((ids[:max_length], segments[:max_length], turn))
        history.append({"role": "assistant", "content": final})
    return turns, missing, full_lengths


def load_sidecar(run_dir: Path, tag: str) -> dict[str, dict[int, str]]:
    sidecar: dict[str, dict[int, str]] = {}
    for path in run_dir.glob(f"{tag}_*_reasoning.jsonl"):
        for line in path.open():
            entry = json.loads(line)
            sidecar.setdefault(entry["example_id"], {})[int(entry["turn"])] = entry["reasoning"]
    return sidecar


def dry_run(args, tokenizer, family: str, chat_kwargs: dict) -> None:
    for seed in args.seeds:
        run_dir = args.run / f"seed{seed}"
        records = [json.loads(line) for line in (run_dir / f"{args.tag}_impossible_original.jsonl").open()][: args.n]
        sidecar = load_sidecar(run_dir, args.tag)
        lengths, counts, missing_total, shown = [], {1: 0, 2: 0}, 0, False
        for record in records:
            example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
            turns, missing, full_lengths = episode_turns(tokenizer, family, example, record, sidecar.get(record["example_id"], {}), args.max_length, chat_kwargs)
            lengths.extend(full_lengths); missing_total += len(missing)
            for ids, segments, _ in turns:
                for segment in counts:
                    counts[segment] += int((segments == segment).sum())
                if not shown and (segments == 1).any() and (segments == 2).any():
                    first_reasoning, first_answer = int(np.argmax(segments == 1)), int(np.argmax(segments == 2))
                    last_answer = len(segments) - 1 - int(np.argmax(segments[::-1] == 2))
                    print(f"seed {seed} boundaries: ...{tokenizer.decode(ids[first_reasoning - 12:first_reasoning])!r} | reasoning {tokenizer.decode(ids[first_reasoning:first_reasoning + 8])!r} ... "
                          f"{tokenizer.decode(ids[first_answer - 12:first_answer])!r} | answer {tokenizer.decode(ids[first_answer:first_answer + 8])!r} ... "
                          f"{tokenizer.decode(ids[last_answer - 4:last_answer + 1])!r} | {tokenizer.decode(ids[last_answer + 1:])!r}")
                    shown = True
        lengths_array = np.asarray(lengths)
        print(f"seed {seed}: {len(records)} episodes, {len(lengths)} turns, {int(lengths_array.sum())} tokens (reasoning {counts[1]}, answer {counts[2]} after the cap), "
              f"turn length median {int(np.median(lengths_array))} / p95 {int(np.percentile(lengths_array, 95))} / max {int(lengths_array.max())}, "
              f"over the {args.max_length} cap {int((lengths_array > args.max_length).sum())}, missing reasoning turns {missing_total}")


class TurnSums:
    """Token sums of one forward pass, per (segment, layer); merged into the episode only when the pass completes."""

    def __init__(self, L: int, hidden: int, K: int, named: int, device):
        self.n = {s: 0 for s in SEGMENTS}
        self.sum = {s: torch.zeros(L, hidden, dtype=torch.float32, device=device) for s in SEGMENTS}
        self.norm = {s: torch.zeros(L, dtype=torch.float64, device=device) for s in SEGMENTS}
        self.proj = {s: torch.zeros(L, K, dtype=torch.float64, device=device) for s in SEGMENTS}
        self.square = {s: torch.zeros(L, K, dtype=torch.float64, device=device) for s in SEGMENTS}
        self.cross = {s: torch.zeros(L, named, K, dtype=torch.float64, device=device) for s in SEGMENTS}
        self.named = named

    def add(self, li: int, hidden_states: torch.Tensor, masks: dict[str, torch.Tensor], dmat: torch.Tensor) -> None:
        for s, mask in masks.items():
            self.n[s] = int(mask.sum())
            if not self.n[s]:
                continue
            hm = hidden_states[mask].float()
            self.sum[s][li] = hm.sum(0)
            self.norm[s][li] = hm.norm(dim=-1).double().sum()
            projection = (hm @ dmat).double()
            self.proj[s][li] = projection.sum(0)
            self.square[s][li] = (projection * projection).sum(0)
            self.cross[s][li] = projection[:, :self.named].T @ projection


def forward_turn(runner, runner_kwargs: dict, layers_module, layers: list[int], ids: list[int], segments: np.ndarray, dmat: dict[int, torch.Tensor],
                 hidden: int, K: int, named: int, device) -> tuple[TurnSums, int]:
    """Run one turn, shrinking it to its first 80% of tokens when the GPU runs out of memory. -> (sums, tokens kept)."""
    keep = len(ids)
    while True:
        sums = TurnSums(len(layers), hidden, K, named, device)
        segment_tensor = torch.as_tensor(segments[:keep], device=device)
        masks = {"reasoning": segment_tensor == 1, "answer": segment_tensor == 2}
        handles = []
        for li, layer in enumerate(layers):
            def hook(_module, _inputs, output, li=li, layer=layer):
                states = output[0] if isinstance(output, tuple) else output
                sums.add(li, states[0].detach(), masks, dmat[layer])
            handles.append(layers_module[layer].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                runner(input_ids=torch.tensor([ids[:keep]], device=device), **runner_kwargs)
            return sums, keep
        except torch.OutOfMemoryError:
            del sums
            torch.cuda.empty_cache()
            if keep < 4096:
                raise
            keep = int(keep * 0.8)
            print(f"out of memory: turn retried on its first {keep} of {len(ids)} tokens", file=sys.stderr, flush=True)
        finally:
            for handle in handles:
                handle.remove()


def main() -> None:
    args = parse_args()
    specifics = steering_model_specifics(args.model)
    family = specifics.family
    chat_kwargs = chat_template_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision or "main")
    if family == "qwen3.5":
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, **chat_kwargs)
        if not prompt_text.endswith("<think>\n"):
            raise SystemExit(f"Qwen generation prompt does not open the thought under {chat_kwargs}: ...{prompt_text[-40:]!r}")
    if args.dry_run:
        dry_run(args, tokenizer, family, chat_kwargs)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    num_layers, hidden = hack_lens.model_dims(args.model)
    layers = args.record_layers or list(range(num_layers))
    L = len(layers)
    device = torch.device("cuda:0")
    directions: dict[str, torch.Tensor] = {}
    for spec in args.direction:
        name, matrix = hack_lens.load_direction_matrix(spec, num_layers, hidden)
        directions[name] = matrix
    named = len(directions)
    nulls = np.stack([random_directions(hidden, args.null_directions, NULL_SEED, layer) for layer in range(num_layers)]) if args.null_directions else None
    for k in range(args.null_directions):
        directions[f"null_{k}"] = torch.as_tensor(nulls[:, k])
    names = list(directions)
    K = len(names)
    dmat = {layer: torch.stack([directions[n][layer] for n in names], dim=1).to(device) for layer in layers}  # (hidden, K)

    print("loading model", file=sys.stderr, flush=True)
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto", "local_files_only": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if attention := (args.attn_implementation or specifics.hf_attn_implementation):
        load_kwargs["attn_implementation"] = attention
    model = hack_lens.load_causal_lm(args.model, **load_kwargs).eval()
    layers_module = hack_lens.get_decoder_layers(model)
    assert len(layers_module) == num_layers, (len(layers_module), num_layers)
    # The base model (no LM head): logits over a 70k-token turn would not fit, and only the residual stream is read.
    base = getattr(model, model.base_model_prefix, None)
    runner, runner_kwargs = (base, {"use_cache": False}) if isinstance(base, torch.nn.Module) and base is not model else (model, {"use_cache": False, "logits_to_keep": 1})

    for seed in args.seeds:
        out_path = args.out / f"{args.arm}__seed{seed}.pt"
        if out_path.exists():
            print(f"{out_path.name}: exists, skipped", file=sys.stderr, flush=True)
            continue
        run_dir = args.run / f"seed{seed}"
        records = [json.loads(line) for line in (run_dir / f"{args.tag}_impossible_original.jsonl").open()][: args.n]
        sidecar = load_sidecar(run_dir, args.tag)
        E = len(records)
        store = {
            "keys": [], "example_id": [], "verdict_key": [], "passed": [], "passed_original": [], "n_attempts": [],
            "turn_lengths": [], "truncated_turns": [], "oom_truncated": [], "missing_reasoning": [],
            "reasoning_mean": torch.zeros(E, L, hidden, dtype=torch.float32), "answer_mean": torch.zeros(E, L, hidden, dtype=torch.float32),
            "stats": {s: {"n": np.zeros(E, np.int64), "h_norm": np.full((E, L), np.nan, np.float64), "proj": np.full((E, L, K), np.nan, np.float64),
                          "proj_square_sum": np.zeros((E, L, K), np.float64), "proj_cross_sum": np.zeros((E, L, named, K), np.float64)} for s in SEGMENTS},
        }
        t0 = time.time()
        for e, record in enumerate(records):
            example = make_dossiers.find_example(args.data, args.tag, record["example_id"], run_dir)
            turns, missing, full_lengths = episode_turns(tokenizer, family, example, record, sidecar.get(record["example_id"], {}), args.max_length, chat_kwargs)
            store["keys"].append(f"{args.arm}/seed{seed}/{record['example_id']}")
            store["example_id"].append(record["example_id"])
            store["verdict_key"].append(f"{args.run.name}/seed{seed}/{record['example_id']}")
            store["passed"].append(bool(record.get("passed"))); store["passed_original"].append(bool(record.get("passed_original")))
            store["n_attempts"].append(int(record.get("n_attempts") or len(record["attempts"])))
            store["turn_lengths"].append(full_lengths); store["truncated_turns"].append([t for t, n in enumerate(full_lengths) if n > args.max_length])
            store["missing_reasoning"].append(missing)
            totals = TurnSums(L, hidden, K, named, device)
            oom = []
            for ids, segments, turn in turns:
                sums, kept = forward_turn(runner, runner_kwargs, layers_module, layers, ids, segments, dmat, hidden, K, named, device)
                if kept < len(ids):
                    oom.append((turn, kept))
                for s in SEGMENTS:
                    totals.n[s] += sums.n[s]
                    for field in ("sum", "norm", "proj", "square", "cross"):
                        getattr(totals, field)[s] += getattr(sums, field)[s]
                del sums
            store["oom_truncated"].append(oom)
            for s in SEGMENTS:
                n = max(totals.n[s], 1)
                stats = store["stats"][s]
                stats["n"][e] = totals.n[s]
                store[f"{s}_mean"][e] = (totals.sum[s] / n).cpu()
                if totals.n[s]:
                    stats["h_norm"][e] = (totals.norm[s] / n).cpu().numpy(); stats["proj"][e] = (totals.proj[s] / n).cpu().numpy()
                stats["proj_square_sum"][e] = totals.square[s].cpu().numpy(); stats["proj_cross_sum"][e] = totals.cross[s].cpu().numpy()
            del totals
            if e % 10 == 0 or e == E - 1:
                print(f"{args.arm} seed {seed}: {e + 1}/{E} episodes ({len(turns)} turns, {sum(full_lengths)} tokens), {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        for s in SEGMENTS:  # fp16 unless a coordinate would overflow it
            if float(store[f"{s}_mean"].abs().max()) < 3e4:
                store[f"{s}_mean"] = store[f"{s}_mean"].to(torch.float16)
        store.update({"arm": args.arm, "seed": seed, "run": str(args.run), "tag": args.tag, "model": args.model, "revision": args.revision, "family": family,
                      "chat_kwargs": chat_kwargs, "record_layers": layers, "directions": names, "named_directions": names[:named],
                      "direction_specs": args.direction, "null_seed": NULL_SEED, "max_length": args.max_length, "seconds": time.time() - t0})
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(store, temporary)
        temporary.rename(out_path)
        print(f"wrote {out_path} ({E} episodes, {store['seconds']:.0f} s, truncated turns {sum(len(t) for t in store['truncated_turns'])}, "
              f"out-of-memory retries {sum(len(o) for o in store['oom_truncated'])}, missing reasoning turns {sum(len(m) for m in store['missing_reasoning'])})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
