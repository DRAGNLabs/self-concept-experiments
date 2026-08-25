
from typing import Protocol, runtime_checkable

from selfconcept.common.conversation_types import Conversation


@runtime_checkable
class BatchEngine(Protocol):
    """Minimal interface the persona-drift loop needs from a generation engine.
    """

    model_name: str

    def generate_batch(self, conversations: list[Conversation]) -> list[str]: ...

