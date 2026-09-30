"""Correlate CoT-ness or assistant-axis projections with existing benchmark outcomes."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
from typing import get_args

import numpy as np
import sklearn
import torch
import transformers
from transformers import AutoTokenizer

from selfconcept.common.loading import load_causal_lm
from selfconcept.common.paths import REPO_ROOT
from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.evaluate import SUFFIXES
from selfconcept.assistant_axis.projection import build_axis_measurement
from selfconcept.cotness.probe import build_cotness_measurement
from selfconcept.measurement.interface import MeasurementName
from selfconcept.measurement.templates import MODELS, ModelSpec, template_kwargs
from .scenarios import evaluate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", choices=get_args(MeasurementName), default="cotness",
                        help="One measurement per run (default: cotness)")
    parser.add_argument("--model", required=True, help="Registered model key, or HF model ID in assistant-axis mode")
    parser.add_argument("--family", choices=("qwen", "gemma", "muse", "olmo"), help="Required for unregistered HF models")
    parser.add_argument("--revision", help="Checkpoint revision for an unregistered HF model")
    parser.add_argument("--assistant-axis", type=Path, help="Model-matched axis .pt from the assistant-axis pipeline")
    parser.add_argument("--axis-layers", type=int, nargs="+", help="Zero-based decoder layers; defaults to the fixed midpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "experiments/cotness/data/neutral.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--code-n", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--code-offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--probe-source", type=Path, help="Frozen validated probe directory with source_manifest.json")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--code-max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--suffix", choices=SUFFIXES, default="i_would")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--scenarios", nargs="+", help="Defaults to ImpossibleBench for assistant-axis, all scenarios for cotness")
    args = parser.parse_args(argv)
    if args.measurement == "assistant-axis" and (not args.assistant_axis or args.probe_source or args.train_only):
        parser.error("--measurement assistant-axis requires --assistant-axis and excludes --probe-source/--train-only")
    if args.measurement == "cotness" and (args.assistant_axis or args.axis_layers):
        parser.error("Axis options require --measurement assistant-axis; measurements cannot be combined")
    if args.axis_layers and not args.assistant_axis:
        parser.error("--axis-layers requires --assistant-axis")
    if args.model not in MODELS and (args.measurement != "assistant-axis" or not args.family):
        parser.error("Unregistered models require --measurement assistant-axis and --family")
    if args.model in MODELS and (args.family or args.revision):
        parser.error("Registered models already pin their family and revision")
    if min(args.n, args.code_n, args.max_new_tokens, args.code_max_new_tokens, args.max_attempts) < 1:
        parser.error("Counts and budgets must be positive")
    if min(args.offset, args.code_offset, args.temperature, args.top_k) < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid offset or sampling configuration")
    if args.axis_layers is not None and (min(args.axis_layers) < 0 or len(set(args.axis_layers)) != len(args.axis_layers)):
        parser.error("Axis layers must be unique nonnegative indices")
    if args.scenarios is None:
        args.scenarios = (["impossible_original", "impossible_conflicting", "impossible_oneoff"]
                          if args.measurement == "assistant-axis" else
                          ["main", "main_mirrored", "treasure_hunt", "treasure_hunt_mirrored",
                           "perspectives", "perspectives_mirrored", "roleplaying", "insider_trading", "sandbagging",
                           "impossible_original", "impossible_conflicting", "impossible_oneoff", "evilgenie"])
    return args


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.manual_seed(1729)
    spec = MODELS.get(args.model) or ModelSpec(args.model, args.revision or "main", args.family, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model": spec.__dict__, "chat_kwargs": template_kwargs(spec.family),
                "corpus_sha256": None if args.measurement == "assistant-axis" else hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "sklearn": sklearn.__version__, "numpy": np.__version__,
                "python": sys.version, "stage": "started"}
    manifest_path = args.out / "manifest.json"
    if args.assistant_axis:
        manifest["axis_sha256"] = hashlib.sha256(args.assistant_axis.read_bytes()).hexdigest()
    if args.probe_source:
        source = json.loads((args.probe_source / "source_manifest.json").read_text())
        if source["model"] != manifest["model"] or source["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("Frozen probe model/corpus differs from evaluation")
        if not json.loads((args.probe_source / "validation.json").read_text())["usable"]:
            raise ValueError("Cannot reuse a probe that failed validation")
        manifest["probe_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.probe_source.iterdir() if p.is_file()}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        # New optional measurements preserve resumability of legacy CoT runs.
        for key in ("family", "revision", "assistant_axis", "axis_layers", "measurement"):
            previous["args"].setdefault(key, "cotness" if key == "measurement" else None)
        if any(previous.get(k) != manifest.get(k) for k in ("args", "model", "chat_kwargs", "corpus_sha256", "probe_sha256", "axis_sha256")):
            raise ValueError("Refusing to mix different configurations in an output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    tok = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision, local_files_only=True)
    model = load_causal_lm(spec.model, revision=spec.revision, local_files_only=True, dtype=torch.bfloat16,
                           device_map="auto")
    model.eval()
    count = len(get_decoder_layers(model))
    layers = [int(count * fraction)-1 for fraction in (.25, .5, .75)]
    if args.measurement == "assistant-axis":
        axis_layers = args.axis_layers if args.axis_layers is not None else [layers[1]]
        config = getattr(model.config, "text_config", model.config)
        measurement = build_axis_measurement(args.assistant_axis, axis_layers, count, config.hidden_size, spec.model)
        frozen_axis = args.out / "assistant_axis.pt"
        if args.assistant_axis.resolve() != frozen_axis.resolve():
            shutil.copy2(args.assistant_axis, frozen_axis)
        manifest["assistant_axis"] = {"sha256": manifest["axis_sha256"], "layers": axis_layers,
                                      "primary_layer": axis_layers[0], "artifact": frozen_axis.name,
                                      "site": "post_decoder_layer_residual", "normalization": "unit_direction"}
        report = None
    else:
        measurement, report = build_cotness_measurement(model, tok, spec.family, args.corpus, args.probe_source,
                                                        args.out / "probes", layers)
        manifest["probe_usable"] = report["usable"]
    if report is None or report["usable"]:
        manifest["measurement"] = {"name": measurement.name, "layers": measurement.layers,
                                   "primary_layer": measurement.primary_layer}
    manifest["stage"] = "axis_ready" if report is None else "probe_ready" if report["usable"] else "probe_failed_validation"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if report is not None and not report["usable"]:
        print("Midpoint probe failed validation; no deception correlations will be interpreted.", flush=True)
        return
    if not args.train_only:
        evaluate(args, model, tok, measurement, family=spec.family)
    manifest["stage"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
