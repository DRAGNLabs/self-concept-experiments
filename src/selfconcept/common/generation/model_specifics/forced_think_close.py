from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, HFStyleAssistantTurn, TemplateWithoutThinkingFlag
from selfconcept.common.hf_strong_types import HFTokenizer, configure_call


class ForcedThinkCloseGenerationSpecifics(TemplateWithoutThinkingFlag, HFStyleAssistantTurn, GenerationSpecifics):
    def direct_answer_prefix_ids(self, tokenizer: HFTokenizer) -> list[int]:
        # OLMo 3's template always opens "<think>" and ignores enable_thinking.
        return configure_call(tokenizer)("</think>\n\n", add_special_tokens=False)["input_ids"]
