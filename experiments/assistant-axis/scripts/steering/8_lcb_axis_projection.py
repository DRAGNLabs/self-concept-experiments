"""Project LiveCodeBench transcripts onto the assistant axis, token by token.

Where does a coding transcript sit on the assistant axis, and does a persona system prompt
move it? For every record in the given 7_steered_lcb.py output files, rebuild the exact
generation input (chat template of the record's system prompt -- or the model's default --
and user message, then the completion), run it through the model once without steering, and
at every decoder layer project each token's residual output onto that layer's unit axis
direction. The projections are split into segments by character span: the system prompt's
content, the user message, the thinking (up to and including ``</think>``), and the final
answer; ``last_prompt`` is the last token before generation starts.

Scale: the axis is mean(default-role activations) - mean(other roles'), so its raw norm at a
layer is the gap between the Assistant and the average role on the axis-extraction data --
the natural unit for a shift in projection. Records that ran steered are re-read unsteered,
so their projections describe the text they produced, not the steering vector.

Writes ``{stem}_axisproj.pt`` next to each input: per record, the per-segment mean projection
at every layer, the last-prompt-token projection at every layer, and the full per-token trace
at the target layer. Files already projected are skipped.

Usage (from experiments/assistant-axis):
    python scripts/steering/8_lcb_axis_projection.py --run.axis_path AXIS --run.inputs '[OUT/coef_+0.00.jsonl, ...]'
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from selfconcept.assistant_axis.internals.model import ProbingModel
from selfconcept.assistant_axis.models import get_config
from selfconcept.common.hf_utils import build_conversation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16}
SEGMENTS = ("system", "user", "thinking", "answer")
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class RunConfig:
    model: str = "allenai/Olmo-3.1-32B-Think"
    axis_path: Path = Path("axis_response_only.pt")
    inputs: list[Path] = field(default_factory=list)
    target_layer: int | None = None
    max_length: int = 24576
    dtype: str = "bfloat16"


def unit_axes(path: Path) -> Float[Tensor, "layers hidden"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    axis = (data["axis"] if isinstance(data, dict) else data).float()
    return axis / (axis.norm(dim=1, keepdim=True) + 1e-8)


def char_spans(record: dict, prompt_text: str) -> dict[str, tuple[int, int]]:
    """Character span of each segment in prompt_text + completion."""
    spans = {}
    if record.get("system_prompt"):
        start = prompt_text.find(record["system_prompt"])
        spans["system"] = (start, start + len(record["system_prompt"]))
    start = prompt_text.find(record["prompt"])
    spans["user"] = (start, start + len(record["prompt"]))
    end_prompt, completion = len(prompt_text), record["completion"]
    close = completion.find(THINK_CLOSE)
    end_think = end_prompt + (close + len(THINK_CLOSE) if close >= 0 else len(completion))
    spans["thinking"] = (end_prompt, end_think)
    spans["answer"] = (end_think, end_prompt + len(completion))
    assert all(a >= 0 for a, _ in spans.values()), "segment text not found in the rebuilt prompt"
    return spans


def project_record(probing_model: ProbingModel, units: Tensor, record: dict, run: RunConfig, target_layer: int) -> dict:
    tok = probing_model.tokenizer
    prompt_text = tok.apply_chat_template(build_conversation(record["prompt"], record.get("system_prompt"), tok),
                                          tokenize=False, add_generation_prompt=True)
    spans = char_spans(record, prompt_text)
    enc = tok(prompt_text + record["completion"], add_special_tokens=False, return_offsets_mapping=True,
              truncation=True, max_length=run.max_length)
    starts = np.array([a for a, _ in enc["offset_mapping"]])
    n_prompt = int((starts < len(prompt_text)).sum())

    layers = probing_model.get_layers()
    projections: list[Tensor | None] = [None] * len(layers)

    def hook_for(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            projections[index] = (hidden[0].float() @ units[index]).cpu()
        return hook

    handles = [layer.register_forward_hook(hook_for(i)) for i, layer in enumerate(layers)]
    try:
        ids = torch.tensor([enc["input_ids"]], device=probing_model.device)
        with torch.no_grad():
            probing_model.model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    per_layer: Float[Tensor, "layers tokens"] = torch.stack(projections)

    segment_means, segment_tokens = {}, {}
    for name in SEGMENTS:
        if name not in spans:
            continue
        a, b = spans[name]
        mask = torch.from_numpy((starts >= a) & (starts < b))
        segment_tokens[name] = int(mask.sum())
        segment_means[name] = per_layer[:, mask].mean(dim=1).numpy() if mask.any() else None
    return {
        "example_id": record["example_id"], "persona": record.get("persona"), "system_prompt": record.get("system_prompt"),
        "coefficient": record["coefficient"], "passed": record["passed"], "n_tokens": len(starts), "n_prompt_tokens": n_prompt,
        "truncated_input": len(starts) < len(tok(prompt_text + record["completion"], add_special_tokens=False)["input_ids"]),
        "segment_tokens": segment_tokens, "segment_means": segment_means,
        "last_prompt": per_layer[:, n_prompt - 1].numpy(),
        "trace": per_layer[target_layer, n_prompt:].numpy().astype(np.float16),  # generated tokens only
    }


def main(run: RunConfig = RunConfig()) -> None:
    probing_model = ProbingModel(run.model, dtype=DTYPE_MAP[run.dtype])
    target_layer = run.target_layer if run.target_layer is not None else get_config(run.model)["target_layer"]
    raw = torch.load(run.axis_path.expanduser(), map_location="cpu", weights_only=False)
    raw = (raw["axis"] if isinstance(raw, dict) else raw).float()
    units = unit_axes(run.axis_path.expanduser()).to(probing_model.device)
    assert units.shape[0] == len(probing_model.get_layers())
    for path in run.inputs:
        out = path.with_name(f"{path.stem}_axisproj.pt")
        if out.exists():
            logger.info("skip %s (done)", out)
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        projected = [project_record(probing_model, units, r, run, target_layer)
                     for r in tqdm(records, desc=path.stem)]
        torch.save({"source": str(path), "model": run.model, "axis_path": str(run.axis_path), "target_layer": target_layer,
                    "axis_raw_norm": raw.norm(dim=1).numpy(), "records": projected}, out)
        at = [p["segment_means"]["thinking"][target_layer] for p in projected if p["segment_means"].get("thinking") is not None]
        logger.info("%s: %d records, mean thinking projection at L%d %.3f", out.name, len(projected), target_layer, float(np.mean(at)))


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
