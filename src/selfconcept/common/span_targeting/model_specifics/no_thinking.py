from dataclasses import dataclass

from selfconcept.common.hf_strong_types import HFTokenizer
from selfconcept.common.span_targeting.model_specifics._shared import decode_token_pieces, place_after_prompt, trim_span
from selfconcept.common.span_targeting.types import CompletionSpans, PromptCompletion, TokenSpan


@dataclass(frozen=True)
class NoThinkingModelSpecifics:
    tokenizer: HFTokenizer

    def split_completion(self, example: PromptCompletion) -> CompletionSpans:
        completion_token_ids = example.completion_token_ids
        response_span = trim_span(
            TokenSpan(0, len(completion_token_ids)),
            completion_token_ids,
            decode_token_pieces(self.tokenizer, completion_token_ids),
            set(self.tokenizer.all_special_ids),
        )
        return CompletionSpans(thinking=[], response=place_after_prompt([response_span], example))
