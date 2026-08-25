
import logging
from typing import NotRequired, Optional, TypedDict, Unpack

from selfconcept.common.conversation_types import Conversation
from selfconcept.transcript_generation.generation import BatchEngine, format_conversation

logger = logging.getLogger(__name__)

class VLLMGeneratorArgs(TypedDict):
    model_name: str
    max_model_len: NotRequired[int]
    tensor_parallel_size: NotRequired[Optional[int]]
    gpu_memory_utilization: NotRequired[float]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    dtype: NotRequired[str]
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
            dtype: Model weight dtype passed to vLLM ("auto", "bfloat16", "float16", ...).
                Use "float16" on GPUs without bf16 support (e.g. V100).
        """
        self.model_name = args["model_name"]
        self.max_model_len = args.get("max_model_len", 2048)
        self.tensor_parallel_size = args.get("tensor_parallel_size", None)
        self.gpu_memory_utilization = args.get("gpu_memory_utilization", 0.9)
        self.temperature = args.get("temperature", 0.7)
        self.max_tokens = args.get("max_tokens", 512)
        self.top_p = args.get("top_p", 0.9)
        self.dtype = args.get("dtype", "auto")
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
        self.load()

        tokenizer = self.llm.get_tokenizer()
        max_len = self.llm.model_config.max_model_len

        # Disable thinking for Qwen models (https://github.com/vllm-project/vllm/issues/18066)
        chat_template_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_template_kwargs["enable_thinking"] = False

        # A prompt longer than the context window makes llm.generate raise for the
        # whole batch. Skip those (return "" for them) so one over-long prompt can't
        # abort every other conversation in the batch.
        prompts = []
        keep_indices = []
        for idx, conv in enumerate(conversations):
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs
            )
            if len(tokenizer.encode(prompt, add_special_tokens=False)) >= max_len:
                logger.warning("Skipping prompt %d: exceeds context window (%d tokens)", idx, max_len)
                continue
            prompts.append(prompt)
            keep_indices.append(idx)

        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        responses = [""] * len(conversations)
        for output, idx in zip(outputs, keep_indices):
            responses[idx] = output.outputs[0].text
        return responses

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
        results = []
        for conv, meta, response in zip(all_conversations, all_metadata, responses):
            result = {
                "system_prompt": meta["system_prompt"],
                "prompt_index": meta["prompt_index"],
                "question_index": meta["question_index"],
                "question": meta["question"],
                "conversation": conv + [{"role": "assistant", "content": response}],
            }
            results.append(result)

        return results


