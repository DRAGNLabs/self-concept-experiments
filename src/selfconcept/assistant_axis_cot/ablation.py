import numpy as np
from jaxtyping import Int8

from selfconcept.assistant_axis_cot.records import SEQUENCE_REGIONS
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion

ATTENTION_SINK_TOKEN_COUNT = 1


def sequence_region_codes(
    example: PromptCompletion,
    completion_spans: CompletionSpans,
    max_length: int,
) -> Int8[np.ndarray, " n_tokens"]:
    """Per token of the concatenated prompt and completion, up to max_length, an index into SEQUENCE_REGIONS;
    the attention sink and the completion's delimiters are unlabelled."""
    sequence_length = len(example.prompt_token_ids) + len(example.completion_token_ids)
    region_codes = np.full(sequence_length, SEQUENCE_REGIONS.index("unlabelled"), dtype=np.int8)
    region_codes[ATTENTION_SINK_TOKEN_COUNT : len(example.prompt_token_ids)] = SEQUENCE_REGIONS.index("prompt")
    for region, spans in (("cot", completion_spans.thinking), ("final", completion_spans.response)):
        for span in spans:
            region_codes[span.start : span.end] = SEQUENCE_REGIONS.index(region)
    return region_codes[:max_length]
