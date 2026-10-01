"""Correlate CoT-ness or assistant-axis projections with existing benchmark outcomes."""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import cast, get_args

import numpy as np
import sklearn
import torch
import transformers
from transformers import AutoTokenizer, PreTrainedModel

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.jsonl import read_jsonl
from selfconcept.common.loading import load_causal_lm
from selfconcept.common.paths import REPO_ROOT
from selfconcept.soo.activations import get_decoder_layers
from selfconcept.soo.evaluate import SUFFIXES
from selfconcept.assistant_axis.projection import build_axis_measurement
from selfconcept.cotness.probe import build_cotness_measurement
from selfconcept.measurement.interface import Measurement, MeasurementName
from selfconcept.measurement.templates import MODEL_SPECS_BY_KEY, ModelFamily, ModelSpec, reasoning_template_kwargs
from .config import RunConfig
from .generate import MeasuredModel
from .records import Manifest, TranscriptRecord
from .scenarios import run_scenarios
from .transcripts import score_transcripts

logger = logging.getLogger(__name__)

AXIS_DEFAULT_SCENARIOS = ["impossible_original", "impossible_conflicting", "impossible_oneoff"]
COTNESS_DEFAULT_SCENARIOS = ["main", "main_mirrored", "treasure_hunt", "treasure_hunt_mirrored",
                             "perspectives", "perspectives_mirrored", "roleplaying", "insider_trading", "sandbagging",
                             "impossible_original", "impossible_conflicting", "impossible_oneoff", "evilgenie"]
# gpt-oss has no sdpa path and eager attention materializes every score (~600 GB at 70k tokens); this flash
# attention kernel supports its attention sinks on Hopper GPUs.
ATTENTION_IMPLEMENTATION_BY_FAMILY: dict[ModelFamily, str] = {"gpt-oss": "kernels-community/vllm-flash-attn3"}
RESUME_IDENTITY_KEYS = ("args", "model", "chat_kwargs", "corpus_sha256", "probe_sha256", "axis_sha256",
                        "transcripts_sha256")


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", choices=get_args(MeasurementName), default="cotness",
                        help="One measurement per run (default: cotness)")
    parser.add_argument("--model", required=True, help="Registered model key, or HF model ID in assistant-axis mode")
    parser.add_argument("--family", choices=("qwen", "gemma", "muse", "olmo", "gpt-oss"), help="Required for unregistered HF models")
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
    parser.add_argument("--transcripts", type=Path,
                        help="Score this transcripts.jsonl (TranscriptRecord lines) instead of generating")
    args = parser.parse_args(argv)
    if args.measurement == "assistant-axis" and (not args.assistant_axis or args.probe_source or args.train_only):
        parser.error("--measurement assistant-axis requires --assistant-axis and excludes --probe-source/--train-only")
    if args.measurement == "cotness" and (args.assistant_axis or args.axis_layers):
        parser.error("Axis options require --measurement assistant-axis; measurements cannot be combined")
    if args.axis_layers and not args.assistant_axis:
        parser.error("--axis-layers requires --assistant-axis")
    if args.model not in MODEL_SPECS_BY_KEY and (args.measurement != "assistant-axis" or not args.family):
        parser.error("Unregistered models require --measurement assistant-axis and --family")
    if args.model in MODEL_SPECS_BY_KEY and (args.family or args.revision):
        parser.error("Registered models already pin their family and revision")
    if min(args.n, args.code_n, args.max_new_tokens, args.code_max_new_tokens, args.max_attempts) < 1:
        parser.error("Counts and budgets must be positive")
    if min(args.offset, args.code_offset, args.temperature, args.top_k) < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid offset or sampling configuration")
    if args.axis_layers is not None and (min(args.axis_layers) < 0 or len(set(args.axis_layers)) != len(args.axis_layers)):
        parser.error("Axis layers must be unique nonnegative indices")
    if args.transcripts and (args.train_only or args.scenarios):
        parser.error("--transcripts takes its scenarios from the file and excludes --train-only")
    if args.transcripts:
        transcripts: list[TranscriptRecord] = read_jsonl(args.transcripts)
        args.scenarios = list(dict.fromkeys(transcript["scenario"] for transcript in transcripts))
    if args.scenarios is None:
        args.scenarios = AXIS_DEFAULT_SCENARIOS if args.measurement == "assistant-axis" else COTNESS_DEFAULT_SCENARIOS
    return RunConfig(**vars(args))


def resolve_model_spec(config: RunConfig) -> ModelSpec:
    if config.model in MODEL_SPECS_BY_KEY:
        return MODEL_SPECS_BY_KEY[config.model]
    assert config.family is not None, "parse_args requires --family for unregistered models"
    return ModelSpec(config.model, config.revision or "main", config.family, 0)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_manifest(config: RunConfig, spec: ModelSpec) -> Manifest:
    manifest: Manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in config._asdict().items()},
        "model": asdict(spec), "chat_kwargs": reasoning_template_kwargs(spec.family),
        "corpus_sha256": None if config.measurement == "assistant-axis" else file_sha256(config.corpus),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "python": sys.version, "stage": "started"}
    if config.assistant_axis:
        manifest["axis_sha256"] = file_sha256(config.assistant_axis)
    if config.transcripts:
        manifest["transcripts_sha256"] = file_sha256(config.transcripts)
    if config.probe_source:
        source = json.loads((config.probe_source / "source_manifest.json").read_text())
        if source["model"] != manifest["model"] or source["corpus_sha256"] != manifest["corpus_sha256"]:
            raise ValueError("Frozen probe model/corpus differs from evaluation")
        if not json.loads((config.probe_source / "validation.json").read_text())["usable"]:
            raise ValueError("Cannot reuse a probe that failed validation")
        manifest["probe_sha256"] = {path.name: file_sha256(path) for path in config.probe_source.iterdir() if path.is_file()}
    return manifest


def check_resumable(previous: Manifest, manifest: Manifest) -> None:
    # New optional measurements preserve resumability of legacy CoT runs.
    legacy_arg_defaults = {"family": None, "revision": None, "assistant_axis": None, "axis_layers": None,
                           "measurement": "cotness", "transcripts": None}
    previous_args = {**legacy_arg_defaults, **previous["args"]}
    previous_identity = {key: previous.get(key) for key in RESUME_IDENTITY_KEYS} | {"args": previous_args}
    if any(previous_identity[key] != manifest.get(key) for key in RESUME_IDENTITY_KEYS):
        raise ValueError("Refusing to mix different configurations in an output directory")


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def prepare_axis_measurement(config: RunConfig, spec: ModelSpec, model: PreTrainedModel, num_layers: int,
                             default_layer: int, manifest: Manifest) -> tuple[Measurement, Manifest]:
    assert config.assistant_axis is not None and "axis_sha256" in manifest, "parse_args requires --assistant-axis"
    axis_layers = config.axis_layers if config.axis_layers is not None else [default_layer]
    text_config = getattr(model.config, "text_config", model.config)
    measurement = build_axis_measurement(config.assistant_axis, axis_layers, num_layers, text_config.hidden_size, spec.model)
    frozen_axis = config.out / "assistant_axis.pt"
    if config.assistant_axis.resolve() != frozen_axis.resolve():
        shutil.copy2(config.assistant_axis, frozen_axis)
    return measurement, {**manifest, "assistant_axis": {
        "sha256": manifest["axis_sha256"], "layers": axis_layers, "primary_layer": axis_layers[0],
        "artifact": frozen_axis.name, "site": "post_decoder_layer_residual", "normalization": "unit_direction"}}


def main() -> None:
    config = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.manual_seed(1729)
    spec = resolve_model_spec(config)
    config.out.mkdir(parents=True, exist_ok=True)
    manifest_path = config.out / "manifest.json"
    manifest: Manifest = initial_manifest(config, spec)
    if manifest_path.exists():
        check_resumable(json.loads(manifest_path.read_text()), manifest)
    write_manifest(manifest_path, manifest)
    tokenizer = cast(HFTokenizer, AutoTokenizer.from_pretrained(spec.model, revision=spec.revision, local_files_only=True))
    attention_kwargs = ({"attn_implementation": ATTENTION_IMPLEMENTATION_BY_FAMILY[spec.family]}
                        if spec.family in ATTENTION_IMPLEMENTATION_BY_FAMILY else {})
    model: PreTrainedModel = load_causal_lm(spec.model, revision=spec.revision, local_files_only=True,
                                            dtype=torch.bfloat16, device_map="auto", **attention_kwargs)
    model.eval()
    num_layers = len(get_decoder_layers(model))
    probe_layers = [int(num_layers * fraction) - 1 for fraction in (.25, .5, .75)]
    if config.measurement == "assistant-axis":
        measurement, manifest = prepare_axis_measurement(config, spec, model, num_layers, probe_layers[1], manifest)
        stage = "axis_ready"
    else:
        measurement = build_cotness_measurement(model, tokenizer, spec.family, config.corpus, config.probe_source,
                                                config.out / "probes", probe_layers)
        manifest = {**manifest, "probe_usable": measurement is not None}
        stage = "probe_ready" if measurement is not None else "probe_failed_validation"
    if measurement is not None:
        manifest = {**manifest, "measurement": {"name": measurement.name, "layers": measurement.layers,
                                                "primary_layer": measurement.primary_layer}}
    manifest = {**manifest, "stage": stage}
    write_manifest(manifest_path, manifest)
    if measurement is None:
        logger.info("Midpoint probe failed validation; no deception correlations will be interpreted.")
        return
    measured_model = MeasuredModel(model, tokenizer, spec.family, measurement)
    if config.transcripts:
        score_transcripts(config, measured_model)
    elif not config.train_only:
        run_scenarios(config, measured_model)
    write_manifest(manifest_path, {**manifest, "stage": "complete"})


if __name__ == "__main__":
    main()
