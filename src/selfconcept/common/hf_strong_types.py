"""
This file contains (incomplete) utilities for using huggingface in a strongly typed way.
It only supports the features that we've been using so far, so if you need more, feel free
to update this file.

Originally created for transformers 4.57.5, then updated to 5.16.1. Updating transformers
might break things in this file, since there is significant churn in its interfaces.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, overload

from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


# re-exports here, don't remove "unused"
from selfconcept.common.chat import AllRoles, Conversation, ConversationTurn, UserAssistantRoles, _Conversation


type OffsetMappingPresent = list[tuple[int, int]]
type OffsetMappingPossibilities = OffsetMappingPresent | None
class BatchEncoding[OffsetMappingT: OffsetMappingPossibilities](Protocol):
    @overload
    def __getitem__(self, key: Literal["input_ids"]) -> list[int]: ...
    input_ids: list[int]
    @overload
    def __getitem__(self, key: Literal["offset_mapping"]) -> OffsetMappingT: ...


class HFTokenizer[RoleT: AllRoles = AllRoles](Protocol):
    all_special_ids: list[int]

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        # text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        # text_target: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
        # text_pair_target: Optional[
        #     Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]
        # ] = None,
        add_special_tokens: bool = True,
        # padding: Union[bool, str, PaddingStrategy] = False,
        # truncation: Union[bool, str, TruncationStrategy, None] = None,
        # max_length: Optional[int] = None,
        # stride: int = 0,
        # is_split_into_words: bool = False,
        # pad_to_multiple_of: Optional[int] = None,
        # padding_side: Optional[str] = None,
        # return_tensors: Optional[Union[str, TensorType]] = None,
        # return_token_type_ids: Optional[bool] = None,
        # return_attention_mask: Optional[bool] = None,
        # return_overflowing_tokens: bool = False,
        # return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        # return_length: bool = False,
        # verbose: bool = True,
        # **kwargs,
    ) -> BatchEncoding: ...

    def apply_chat_template(
        self,
        conversation: _Conversation[RoleT] | list[_Conversation[RoleT]],
        # tools: Optional[list[Union[dict, Callable]]] = None,
        # documents: Optional[list[dict[str, str]]] = None,
        # chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        # continue_final_message: bool = False,
        tokenize: bool = True,
        # padding: Union[bool, str, PaddingStrategy] = False,
        # truncation: bool = False,
        # max_length: Optional[int] = None,
        # return_tensors: Optional[Union[str, TensorType]] = None,
        # return_dict: bool = False,
        # return_assistant_tokens_mask: bool = False,
        # tokenizer_kwargs: Optional[dict[str, Any]] = None,
        **template_kwargs: Any,
    ) -> str | list[int] | list[str] | list[list[int]]: ...# | BatchEncoding:

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def decode(
        self,
        token_ids: int | list[int], # , np.ndarray, "torch.Tensor"], # TODO: only import for typecheck
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str: ...


class ConfiguredCall[
    RoleT: AllRoles = AllRoles,
    OffsetMappingT: OffsetMappingPossibilities = None
]:
    def __init__(self, hf_tokenizer: HFTokenizer[RoleT], return_offsets_mapping: bool):
        self._hf_tokenizer = hf_tokenizer
        self._return_offsets_mapping = return_offsets_mapping

    @overload
    def return_offsets_mapping(self, return_offsets_mapping: Literal[True]) -> ConfiguredCall[RoleT, OffsetMappingPresent]: ...
    @overload
    def return_offsets_mapping(self, return_offsets_mapping: Literal[False]) -> ConfiguredCall[RoleT, None]: ...
    def return_offsets_mapping(self, return_offsets_mapping: bool) -> ConfiguredCall[RoleT, OffsetMappingPossibilities]:
        return ConfiguredCall[RoleT, OffsetMappingPossibilities](self._hf_tokenizer, return_offsets_mapping)
    
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        add_special_tokens: bool = True,
    ) -> BatchEncoding[OffsetMappingT]:
        return self._hf_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=self._return_offsets_mapping
        )
    
def configure_call[RoleT: AllRoles](hf_tokenizer: HFTokenizer[RoleT]) -> ConfiguredCall[RoleT]:
    return ConfiguredCall[RoleT](hf_tokenizer, return_offsets_mapping=False)


type ResultPossibilities = BatchEncoding | str
class ConfiguredApplyChatTemplate[
    RoleT: AllRoles = AllRoles,
    ResultT: ResultPossibilities = BatchEncoding,
]:
    def __init__(self, hf_tokenizer: HFTokenizer[RoleT], tokenize: bool):
        self._hf_tokenizer = hf_tokenizer
        self._tokenize: bool = tokenize


    @overload
    def tokenize(self, tokenize: Literal[True]) -> ConfiguredApplyChatTemplate[RoleT, BatchEncoding]: ...
    @overload
    def tokenize(self, tokenize: Literal[False]) -> ConfiguredApplyChatTemplate[RoleT, str]: ...
    def tokenize(self, tokenize: bool) -> ConfiguredApplyChatTemplate[RoleT, ResultPossibilities]:
        return ConfiguredApplyChatTemplate[RoleT, ResultPossibilities](self._hf_tokenizer, tokenize)
    
    # @overload
    # def __call__(
    #     self,
    #     conversation: _Conversation[RoleT],
    #     add_generation_prompt: bool = False,
    #     **template_kwargs: Any,
    # ) -> ResultT: ...
    # @overload
    # def __call__(
    #     self,
    #     conversation: list[_Conversation[RoleT]],
    #     add_generation_prompt: bool = False,
    #     **template_kwargs: Any,
    # ) -> Sequence[ResultT]: ... # this return type is not quite right, but not actually needed right now, so leaving unimplemented. I'm leaving the  comments here in case we need a starting point for the future.
    def __call__(
        self,
        conversation: _Conversation[RoleT],
        add_generation_prompt: bool = False,
        **template_kwargs: Any,
    ) -> ResultT:
        return self._hf_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=self._tokenize,
            **template_kwargs # TODO: technically need to filter out tokenize, etc.
        ) # type: ignore


def configure_apply_chat_template[RoleT: AllRoles = AllRoles](hf_tokenizer: HFTokenizer[RoleT]) -> ConfiguredApplyChatTemplate[RoleT]:
    return ConfiguredApplyChatTemplate(hf_tokenizer, tokenize=True)