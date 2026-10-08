# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict, Unpack, cast

from selfconcept.common.generation.model_specifics import get_generation_specifics
from selfconcept.common.hf_strong_types import Conversation, HFTokenizer, configure_apply_chat_template, configure_call
from selfconcept.common.llm_judge import GeneratedResponse
from selfconcept.common.span_targeting.types import PromptCompletion
from selfconcept.transcript_generation.generation import BatchEngine, format_conversation

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

logger = logging.getLogger(__name__)


def chat_prompt_text(
    tokenizer: HFTokenizer,
    model_name: str,
    conversation: Conversation,
    enable_thinking: bool,
    chat_template_kwargs: dict[str, Any],
) -> str:
    model_specifics = get_generation_specifics(model_name)
    direct_answer_prefix = "" if enable_thinking else tokenizer.decode(model_specifics.direct_answer_prefix_ids(tokenizer))
    return configure_apply_chat_template(tokenizer).tokenize(False)(
        conversation,
        add_generation_prompt=True,
        **model_specifics.set_thinking_flag(chat_template_kwargs, enable_thinking),
    ) + direct_answer_prefix



def generated_prompt_completion(
    tokenizer: HFTokenizer,
    model_name: str,
    conversation: Conversation,
    enable_thinking: bool,
    chat_template_kwargs: dict[str, Any],
) -> PromptCompletion:
    """The tokens the final, generated assistant turn was sampled as. That turn is not re-rendered through the chat
    template, since templates such as OLMo 3's, GLM 4.5's and Gemma 4's do not reproduce generated reasoning."""
    prompt_text = chat_prompt_text(tokenizer, model_name, conversation[:-1], enable_thinking, chat_template_kwargs)
    completion_text = get_generation_specifics(model_name).to_completion(conversation[-1])
    tokenize = configure_call(tokenizer)
    return PromptCompletion(
        tokenize(prompt_text, add_special_tokens=False)["input_ids"],
        tokenize(completion_text, add_special_tokens=False)["input_ids"],
    )

class VLLMGeneratorArgs(TypedDict):
    model_name: str
    max_model_len: NotRequired[int]
    tensor_parallel_size: NotRequired[int]
    gpu_memory_utilization: NotRequired[float]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    dtype: NotRequired[ModelDType]
    enable_thinking: NotRequired[bool]
    chat_template_kwargs: NotRequired[dict[str, str | bool]]
    distributed_executor_backend: NotRequired[Optional[str]]


class VLLMGenerator(BatchEngine):
    """
    Generator for batch inference using vLLM.

    Example:
        generator = VLLMGenerator("google/gemma-2-27b-it")
        responses = generator.generate_batch(conversations)
    """

    def __init__(
        self,
        **args: Unpack[VLLMGeneratorArgs]
    ):
        """
        Initialize vLLM generator.

        Args:
            model_name: HuggingFace model name
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs (None for auto-detect)
            gpu_memory_utilization: GPU memory utilization
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
            top_k: Top-k sampling (-1 disables it)
            dtype: Model weight dtype passed to vLLM ("auto", "bfloat16", "float16", ...).
                Use "float16" on GPUs without bf16 support (e.g. V100).
            chat_template_kwargs: Extra apply_chat_template kwargs, e.g. gpt-oss reasoning_effort
        """
        self.model_name = args["model_name"]
        self.max_model_len = args.get("max_model_len", 2048)
        self.tensor_parallel_size = args.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = args.get("gpu_memory_utilization", 0.9)
        self.temperature = args.get("temperature", 0.7)
        self.max_tokens = args.get("max_tokens", 512)
        self.top_p = args.get("top_p", 0.9)
        self.top_k = args.get("top_k", -1)
        self.dtype: ModelDType = args.get("dtype", "auto")
        self.enable_thinking = args.get("enable_thinking", False)
        self.chat_template_kwargs = args.get("chat_template_kwargs", {})
        self.distributed_executor_backend = args.get("distributed_executor_backend", None)

        self.llm = None
        self.sampling_params = None

    def load(self):
        """Load the vLLM model."""
        if self.llm is not None:
            return

        from vllm import LLM, SamplingParams

        logger.info(f"Loading vLLM model: {self.model_name}")

        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            distributed_executor_backend=self.distributed_executor_backend,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            skip_special_tokens=get_generation_specifics(self.model_name).skip_special_tokens,
        )

        logger.info("Model loaded successfully")

    def generate_batch(
        self,
        conversations: list[Conversation],
    ) -> list[str]:
        """
        Generate responses for a batch of conversations.

        Args:
            conversations: List of conversations (each is a list of message dicts)

        Returns:
            List of generated response texts
        """
        return [response["text"] for response in self.generate_responses(conversations)]

    def generate_responses(
        self,
        conversations: list[Conversation],
    ) -> list[GeneratedResponse]:
        """Like generate_batch, plus whether each response hit max_tokens. A skipped over-long prompt
        counts as truncated."""
        self.load()
        assert self.llm is not None

        tokenizer = self.llm.get_tokenizer()
        max_len = self.llm.model_config.max_model_len

        # A prompt longer than the context window makes llm.generate raise for the
        # whole batch. Skip those (return "" for them) so one over-long prompt can't
        # abort every other conversation in the batch.
        prompts = []
        keep_indices = []
        for idx, conv in enumerate(conversations):
            prompt = chat_prompt_text(
                cast(HFTokenizer, tokenizer), self.model_name, conv, self.enable_thinking, self.chat_template_kwargs
            )
            if len(tokenizer.encode(prompt, add_special_tokens=False)) >= max_len: # type: ignore
                logger.warning("Skipping prompt %d: exceeds context window (%d tokens)", idx, max_len)
                continue
            prompts.append(prompt)
            keep_indices.append(idx)

        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        responses: list[GeneratedResponse] = [{"text": "", "reasoning": None, "truncated": True} for _ in conversations]
        for output, idx in zip(outputs, keep_indices):
            completion = output.outputs[0]
            responses[idx] = {"text": completion.text, "reasoning": None, "truncated": completion.finish_reason == "length"}
        return responses

    def next_token_logprobs(self, prompt_texts: list[str], top_logprobs: int) -> list[dict[int, float] | None]:
        """The top next-token logprobs by token id after each prompt text (tokenized without added special
        tokens), or None for a prompt that leaves no room in the context window for that token."""
        self.load()
        assert self.llm is not None
        from vllm import SamplingParams, TokensPrompt

        tokenizer = self.llm.get_tokenizer()
        max_len = self.llm.model_config.max_model_len
        prompt_ids = [tokenizer.encode(prompt_text, add_special_tokens=False) for prompt_text in prompt_texts]
        fitting_indices = [index for index, ids in enumerate(prompt_ids) if len(ids) < max_len]
        if len(fitting_indices) < len(prompt_texts):
            logger.warning("Skipping %d prompts: exceed context window (%d tokens)", len(prompt_texts) - len(fitting_indices), max_len)

        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids[index]) for index in fitting_indices],
            SamplingParams(max_tokens=1, logprobs=top_logprobs, temperature=0),
        )
        logprobs_by_prompt_index: dict[int, dict[int, float]] = {}
        for index, output in zip(fitting_indices, outputs, strict=True):
            first_token_logprobs = output.outputs[0].logprobs
            assert first_token_logprobs is not None
            logprobs_by_prompt_index[index] = {
                token_id: logprob.logprob for token_id, logprob in first_token_logprobs[0].items()
            }
        return [logprobs_by_prompt_index.get(index) for index in range(len(prompt_texts))]

    def generate_for_role(
        self,
        instructions: list[str],
        questions: list[str],
        prompt_indices: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Generate responses for a role across all instruction variants and questions.

        Args:
            instructions: List of system prompt variants
            questions: List of questions
            prompt_indices: Which instruction indices to use (default: all)

        Returns:
            List of result dicts with conversation, prompt_index, question_index
        """
        self.load()
        assert self.llm is not None
        tokenizer = self.llm.get_tokenizer()

        if prompt_indices is None:
            prompt_indices = list(range(len(instructions)))

        # Build all conversations
        all_conversations = []
        all_metadata = []

        for prompt_idx in prompt_indices:
            if prompt_idx >= len(instructions):
                continue

            instruction = instructions[prompt_idx]

            for q_idx, question in enumerate(questions):
                conversation = format_conversation(instruction, question, tokenizer)
                all_conversations.append(conversation)
                all_metadata.append({
                    "system_prompt": instruction,
                    "prompt_index": prompt_idx,
                    "question_index": q_idx,
                    "question": question,
                })

        if not all_conversations:
            return []

        # Generate
        responses = self.generate_batch(all_conversations)

        # Build results
        model_specifics = get_generation_specifics(self.model_name)
        results = []
        for conv, meta, response in zip(all_conversations, all_metadata, responses):
            result = {
                "system_prompt": meta["system_prompt"],
                "prompt_index": meta["prompt_index"],
                "question_index": meta["question_index"],
                "question": meta["question"],
                "conversation": conv + [model_specifics.to_assistant_turn(response)],
            }
            results.append(result)

        return results

@dataclass(frozen=True)
class RayExecutor:
    target_tensor_parallel: int
    auditor_tensor_parallel: int

@dataclass(frozen=True)
class PartitionedExecutor:
    target_gpus: list[int]
    auditor_gpus: list[int]

@dataclass(frozen=True)
class InProcessExecutor:
    tensor_parallel_size: int

type ExecutorOptions = RayExecutor | PartitionedExecutor | InProcessExecutor

class _CommonVLLMGeneratorArgs(TypedDict):
    max_model_len: NotRequired[int]
    gpu_memory_utilization: NotRequired[float]
    dtype: NotRequired[ModelDType]
    distributed_executor_backend: NotRequired[Optional[str]]

def build_vllm_engines(
        target_model: str,
        auditor_model: str,
        gpu_memory_utilization: float,
        max_model_len: int | None = None,
        target_temperature: float = 0.7,
        target_max_tokens: int = 512,
        auditor_temperature: float = 0.8,
        auditor_max_tokens: int = 4096,
        executor_options: ExecutorOptions = InProcessExecutor(tensor_parallel_size=1),
        dtype: ModelDType = "bfloat16"
):
    """Build (target, auditor, cleanup). ``cleanup`` must be called in a finally.

    Three placement modes:
      - ray (--executor ray): both engines live in this process but use vLLM's
        Ray executor, which puts each engine's TP workers in its own placement
        group on disjoint GPUs. This is how two TP>1 engines run concurrently —
        the mp executor deadlocks when a second TP engine inits. TP per engine
        from --target_tp/--auditor_tp.
      - partitioned (--target_gpus + --auditor_gpus): separate processes, one GPU
        set each, pinned via CUDA_VISIBLE_DEVICES (mp executor). Fine when at most
        one engine uses TP>1.
      - in-process (default): both engines share the --tensor_parallel_size GPUs.
    """
    gmu = gpu_memory_utilization

    if isinstance(executor_options, RayExecutor):
        # We already run inside the persistent .venv, so skip Ray's "uv run" env
        # replication — it errors when cwd (pipeline/) != the pyproject dir. Must be
        # set before `import ray` (the constant is read at import time).
        os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
        import ray
        if not ray.is_initialized():
            ray.init()
        common = _CommonVLLMGeneratorArgs(dtype=dtype,
                      gpu_memory_utilization=0.9 if gmu is None else gmu,
                      distributed_executor_backend="ray")
        if max_model_len is not None:
            common["max_model_len"] = max_model_len
        target = VLLMGenerator(model_name=target_model, tensor_parallel_size=executor_options.target_tensor_parallel,
                               temperature=target_temperature, max_tokens=target_max_tokens, **common)
        auditor = VLLMGenerator(model_name=auditor_model, tensor_parallel_size=executor_options.auditor_tensor_parallel,
                                temperature=auditor_temperature, max_tokens=auditor_max_tokens, **common)
        return target, auditor, ray.shutdown

    common = _CommonVLLMGeneratorArgs(gpu_memory_utilization=gmu, dtype=dtype)
    if max_model_len is not None:
        common["max_model_len"] = max_model_len
    if isinstance(executor_options, PartitionedExecutor):
        from selfconcept.transcript_generation.remote_generation import RemoteEngine
        # tensor_parallel_size is inferred from the GPU list inside RemoteEngine.
        target = RemoteEngine(executor_options.target_gpus, model_name=target_model,
                              temperature=target_temperature, max_tokens=target_max_tokens, **common)
        try:
            auditor = RemoteEngine(executor_options.auditor_gpus, model_name=auditor_model,
                                   temperature=auditor_temperature, max_tokens=auditor_max_tokens, **common)
        except BaseException:
            # Don't leave the first worker alive (non-daemon) and hang the job.
            target.close()
            raise

        def cleanup():
            target.close()
            auditor.close()

        return target, auditor, cleanup

    target = VLLMGenerator(model_name=target_model, tensor_parallel_size=executor_options.tensor_parallel_size,
                           temperature=target_temperature, max_tokens=target_max_tokens, **common)
    auditor = VLLMGenerator(model_name=auditor_model, tensor_parallel_size=executor_options.tensor_parallel_size,
                            temperature=auditor_temperature, max_tokens=auditor_max_tokens, **common)
    return target, auditor, lambda: None

