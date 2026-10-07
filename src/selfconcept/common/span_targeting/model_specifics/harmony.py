from dataclasses import dataclass, field
from itertools import groupby

from openai_harmony import HarmonyEncoding, HarmonyEncodingName, Role, StreamableParser, StreamState, load_harmony_encoding

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.model_specifics._shared import decode_token_pieces, place_after_prompt, trim_span
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan


def _content_channel_by_token(encoding: HarmonyEncoding, completion_token_ids: list[int]) -> list[str | None]:
    """The channel of each message-content token; None for headers and message delimiters."""
    parser = StreamableParser(encoding, Role.ASSISTANT)
    content_channel_by_token: list[str | None] = []
    for token_id in completion_token_ids:
        was_in_content = parser.state == StreamState.CONTENT
        parser.process(token_id)
        is_content = was_in_content and parser.state == StreamState.CONTENT
        content_channel_by_token.append(parser.current_channel if is_content else None)
    return content_channel_by_token


def _message_content_spans_by_channel(content_channel_by_token: list[str | None]) -> dict[str, list[TokenSpan]]:
    """Consecutive content tokens are one message's content, since messages are separated by headers."""
    message_content_spans_by_channel: dict[str, list[TokenSpan]] = {}
    for channel, indexed_channels in groupby(enumerate(content_channel_by_token), key=lambda pair: pair[1]):
        if channel is None:
            continue
        token_indices = [token_index for token_index, _ in indexed_channels]
        message_content_spans_by_channel.setdefault(channel, []).append(
            TokenSpan(token_indices[0], token_indices[-1] + 1)
        )
    return message_content_spans_by_channel


@dataclass(frozen=True)
class HarmonyModelSpecifics:
    tokenizer: HFTokenizer
    encoding: HarmonyEncoding = field(
        default_factory=lambda: load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    )

    def split_completion(self, example: PromptCompletion) -> CompletionSpans:
        completion_token_ids = example.completion_token_ids
        message_content_spans_by_channel = _message_content_spans_by_channel(
            _content_channel_by_token(self.encoding, completion_token_ids)
        )
        token_pieces = decode_token_pieces(self.tokenizer, completion_token_ids)
        special_token_ids = set(self.tokenizer.all_special_ids)

        def trimmed(spans: list[TokenSpan]) -> list[TokenSpan]:
            trimmed_spans = [trim_span(span, completion_token_ids, token_pieces, special_token_ids) for span in spans]
            return place_after_prompt(trimmed_spans, example)

        return CompletionSpans(
            thinking=trimmed(message_content_spans_by_channel.get("analysis", [])),
            response=trimmed(message_content_spans_by_channel.get("final", [])[-1:]),
        )
