from selfconcept.common.generation.model_specifics.types import GenerationSpecifics, TemplateWithoutThinkingFlag
from selfconcept.common.harmony import join_harmony_completion, split_harmony_completion
from selfconcept.common.hf_strong_types import AllRoles, ConversationTurn, HFTokenizer, configure_call

EMPTY_ANALYSIS_PREFILL = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


class HarmonyAssistantTurn(ConversationTurn):
    thinking: str


class HarmonyGenerationSpecifics(TemplateWithoutThinkingFlag, GenerationSpecifics[AllRoles, HarmonyAssistantTurn]):
    skip_special_tokens = False

    def direct_answer_prefix_ids(self, tokenizer: HFTokenizer) -> list[int]:
        return configure_call(tokenizer)(EMPTY_ANALYSIS_PREFILL, add_special_tokens=False)["input_ids"]

    def to_assistant_turn(self, completion: str) -> HarmonyAssistantTurn:
        reasoning, final = split_harmony_completion(completion)
        return {"role": "assistant", "content": final, "thinking": reasoning}

    def to_completion(self, assistant_turn: HarmonyAssistantTurn) -> str:
        return join_harmony_completion(assistant_turn["thinking"], assistant_turn["content"])
