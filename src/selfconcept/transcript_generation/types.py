from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """A persona the auditor role-plays, with a globally unique id."""

    persona_id: int
    domain: str
    instructions: str


@dataclass(frozen=True)
class ConversationSeed:
    """One conversation to generate: a persona paired with a generated topic."""

    persona: Persona
    topic_id: int
    topic: str