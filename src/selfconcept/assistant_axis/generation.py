# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Role-based response generation for axis extraction.

Wraps the shared VLLMGenerator to generate responses for every role's system-prompt
variants across a fixed question set, writing one JSONL file per role.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import jsonlines
from tqdm import tqdm

from selfconcept.assistant_axis.models import get_config
from selfconcept.transcript_generation.vllm_generation import VLLMGenerator

if TYPE_CHECKING:
    from vllm.config.model import ModelDType

logger = logging.getLogger(__name__)


class RoleResponseGenerator:
    """
    Generator for role-based model responses using vLLM batch inference.

    Processes role JSON files and generates responses for all roles.

    Example:
        generator = RoleResponseGenerator(
            model_name="google/gemma-2-27b-it",
            roles_dir="data/roles/instructions",
            output_dir="outputs/responses",
            questions_file="data/extraction_questions.jsonl"
        )
        generator.process_all_roles()
    """

    def __init__(
        self,
        model_name: str,
        roles_dir: Path,
        output_dir: Path,
        questions_file: Path,
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 240,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        prompt_indices: list[int] | None = None,
        short_name: str | None = None,
        dtype: ModelDType = "auto",
        enable_thinking: bool = False,
    ):
        """
        Initialize role response generator.

        Args:
            model_name: HuggingFace model name
            roles_dir: Directory containing role JSON files
            output_dir: Output directory for JSONL files
            questions_file: Path to questions JSONL file
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs the single vLLM engine spans
            gpu_memory_utilization: GPU memory utilization
            question_count: Number of questions per role
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
            prompt_indices: Which prompt indices to use (default: 0-4)
            short_name: Short model name for formatting (auto-detected if None)
            enable_thinking: whether to generate thinking tokens for models that support it
        """
        self.model_name = model_name
        self.roles_dir = roles_dir
        self.output_dir = output_dir
        self.questions_file = questions_file
        self.question_count = question_count
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))

        # Get short name for {model_name} placeholder
        if short_name is None:
            config = get_config(model_name)
            self.short_name = config["short_name"]
        else:
            self.short_name = short_name

        self.generator = VLLMGenerator(
            model_name=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            dtype=dtype,
            enable_thinking=enable_thinking,
        )

        self.questions: list[str] | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RoleResponseGenerator with model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_questions(self) -> list[str]:
        """Load questions from JSONL file."""
        if self.questions is not None:
            return self.questions

        questions = []
        with jsonlines.open(self.questions_file, 'r') as reader:
            for entry in reader:
                questions.append(entry['question'])

        self.questions = questions[:self.question_count]
        logger.info(f"Loaded {len(self.questions)} questions")
        return self.questions

    def load_role(self, role_file: Path) -> dict:
        """Load a role JSON file."""
        with open(role_file, 'r') as f:
            return json.load(f)

    def format_instruction(self, instruction: str) -> str:
        """Format instruction, replacing {model_name} placeholder."""
        # filled with a per-model short name (from get_config), so the same role set is
        # reused verbatim across model families.
        return instruction.replace("{model_name}", self.short_name)

    def generate_role_responses(self, role_name: str, role_data: dict) -> list[dict]:
        """Generate responses for a single role."""
        instructions = role_data.get('instruction', [])
        if not instructions:
            return []

        questions = self.load_questions()

        # Get and format instructions
        formatted_instructions = []
        for inst in instructions:
            raw = inst.get('pos', '')
            formatted_instructions.append(self.format_instruction(raw))

        logger.info(f"Processing role '{role_name}' with {len(questions)} questions")

        # Generate
        results = self.generator.generate_for_role(
            instructions=formatted_instructions,
            questions=questions,
            prompt_indices=self.prompt_indices,
        )

        # Add label
        for r in results:
            r["label"] = "pos"

        return results

    def save_responses(self, role_name: str, responses: list[dict]):
        """Save responses to JSONL file."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        with jsonlines.open(output_file, mode='w') as writer:
            for response in responses:
                writer.write(response)
        logger.info(f"Saved {len(responses)} responses to {output_file}")

    def should_skip_role(self, role_name: str) -> bool:
        """Check if role output already exists."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        return output_file.exists()

    def process_all_roles(
        self,
        skip_existing: bool = True,
        roles: list[str] | None = None,
    ):
        """
        Process all roles and generate responses.

        Args:
            skip_existing: Skip roles with existing output files
            roles: Specific role names to process (None for all)
        """
        # Load model
        self.generator.load()
        self.load_questions()

        # Get role files
        role_files = {}
        for file_path in sorted(self.roles_dir.glob("*.json")):
            role_name = file_path.stem
            try:
                role_data = self.load_role(file_path)
                if 'instruction' not in role_data:
                    logger.warning(f"Skipping {role_name}: missing 'instruction' field")
                    continue
                role_files[role_name] = role_data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Found {len(role_files)} role files")

        # Filter
        if roles:
            role_files = {k: v for k, v in role_files.items() if k in roles}

        if skip_existing:
            role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}

        logger.info(f"Processing {len(role_files)} roles")

        # Process
        for role_name, role_data in tqdm(role_files.items(), desc="Processing roles"):
            try:
                responses = self.generate_role_responses(role_name, role_data)
                if responses:
                    self.save_responses(role_name, responses)
            except Exception as e:
                logger.error(f"Error processing {role_name}: {e}")
