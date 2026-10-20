"""Per-token schemer-axis projections of held-out responses (GPU), the input of token_figures.py.

Runs the model once over every held-out-question response of each persona (judge score 3 only) and of the default
role, takes the residual stream at one layer, and projects the assistant tokens of the reasoning-token span (with_cot)
and of the answer-token span (response_only) that the extraction pools onto that region's unit axis. Writes
{out}/{role}.jsonl, one TokenRecord per response.

    python scripts/token_projections.py --root <extraction root> --with-cot-axis <axis_with_cot>/persona_contrast.pt \
        --response-only-axis <axis_response_only>/persona_contrast.pt --layer 17 \
        --model openai/gpt-oss-120b --out <dir> [--attn-implementation kernels-community/vllm-flash-attn3] [--limit-per-role N]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
import torch

from selfconcept.assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel
from selfconcept.assistant_axis.internals.model_specifics import get_model_specifics_by_name
from selfconcept.common.hf_strong_types import Conversation
from selfconcept.measurement.unit_axes import load_unit_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Region = Literal["with_cot", "response_only"]
INCLUDE_THINKING_BY_REGION: dict[Region, bool] = {"with_cot": True, "response_only": False}


class TokenRecord(TypedDict):
    key: str
    question_index: int
    token_ids_by_region: dict[Region, list[int]]
    scores_by_region: dict[Region, list[float]]


class Response(TypedDict):
    key: str
    question_index: int
    conversation: Conversation


def heldout_responses(root: Path, role: str, heldout_ids: set[int], limit: int | None) -> list[Response]:
    scores_by_key = {} if role == "default" else json.loads((root / "scores" / f"{role}.json").read_text())
    responses = []
    for line in (root / "responses" / f"{role}.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = f"{record['label']}_p{record['prompt_index']}_q{record['question_index']}"
        if record["question_index"] in heldout_ids and (role == "default" or scores_by_key.get(key) == 3):
            responses.append({"key": key, "question_index": record["question_index"], "conversation": record["conversation"]})
    return responses[:limit]


def last_assistant_span(spans: list[dict], conversation_id: int) -> tuple[int, int] | None:
    assistant = [span for span in spans if span["conversation_id"] == conversation_id and span["role"] == "assistant"]
    return (assistant[-1]["local_start"], assistant[-1]["local_end"]) if assistant else None


def project_batch(extractor: ActivationExtractor, encoder: ConversationEncoder, responses: list[Response], layer: int,
                  unit_axis_by_region: dict[Region, Float[torch.Tensor, "hidden"]], max_length: int, chat_kwargs: dict) -> list[TokenRecord]:
    """One forward pass for both regions: the regions only differ in which assistant tokens their spans select."""
    conversations = [response["conversation"] for response in responses]
    activations, _ = extractor.batch_conversations(conversations, layer=layer, max_length=max_length, include_thinking=True, **chat_kwargs)
    projections_by_region = {region: (activations[0].float() @ unit_axis.to(activations.device)).cpu() for region, unit_axis in unit_axis_by_region.items()}
    spans_by_region: dict[Region, tuple[list[list[int]], list[dict], dict]] = {region: encoder.build_batch_turn_spans(conversations, include_thinking=include_thinking, **chat_kwargs)
                       for region, include_thinking in INCLUDE_THINKING_BY_REGION.items()}
    records: list[TokenRecord] = []
    for index, response in enumerate(responses):
        record: TokenRecord = {"key": response["key"], "question_index": response["question_index"], "token_ids_by_region": {}, "scores_by_region": {}}
        for region, (full_ids, spans, _) in spans_by_region.items():
            span = last_assistant_span(spans, index)
            if span is None:
                continue
            projections = projections_by_region[region]
            start, end = span[0], min(span[1], projections.shape[1])
            record["token_ids_by_region"][region] = full_ids[index][start:end]
            record["scores_by_region"][region] = [round(score, 3) for score in projections[index, start:end].tolist()]
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--root", type=Path, required=True, help="extraction root with responses/ and scores/")
    parser.add_argument("--with-cot-axis", type=Path, required=True)
    parser.add_argument("--response-only-axis", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=Path, default=here / "data/persona_groups.json")
    parser.add_argument("--questions", type=Path, default=here / "data/questions.jsonl")
    parser.add_argument("--roles", nargs="+", help="default: default + every role in --groups")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=20480)
    parser.add_argument("--limit-per-role", type=int, default=None)
    args = parser.parse_args()

    heldout_ids = {question["id"] for question in map(json.loads, args.questions.read_text().splitlines()) if question["split"] == "heldout"}
    roles = args.roles or ["default", *[role for roles in json.loads(args.groups.read_text()).values() for role in roles]]
    axis_path_by_region: dict[Region, Path] = {"with_cot": args.with_cot_axis, "response_only": args.response_only_axis}
    num_layers, hidden = torch.load(args.with_cot_axis, map_location="cpu", weights_only=True)["axis"].shape
    unit_axis_by_region: dict[Region, torch.Tensor] = {region: torch.as_tensor(load_unit_axes(path, [args.layer], num_layers, hidden, args.model)[args.layer]["direction"])
                           for region, path in axis_path_by_region.items()}

    probing_model = ProbingModel(args.model, attn_implementation=args.attn_implementation)
    assert probing_model.tokenizer is not None
    encoder = ConversationEncoder(probing_model.tokenizer, probing_model.model_name)
    extractor = ActivationExtractor(probing_model, encoder)
    chat_kwargs = get_model_specifics_by_name(args.model).set_thinking_flag({}, True)
    args.out.mkdir(parents=True, exist_ok=True)
    for role in roles:
        out_path = args.out / f"{role}.jsonl"
        if out_path.exists() or not (args.root / "responses" / f"{role}.jsonl").exists():
            continue
        responses = heldout_responses(args.root, role, heldout_ids, args.limit_per_role)
        records = [record for start in range(0, len(responses), args.batch_size)
                   for record in project_batch(extractor, encoder, responses[start:start + args.batch_size], args.layer, unit_axis_by_region,
                                               args.max_length, chat_kwargs)]
        partial_path = out_path.with_suffix(".jsonl.partial")
        partial_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        partial_path.rename(out_path)
        logger.info(f"{role}: {len(records)} responses")


if __name__ == "__main__":
    main()
