from __future__ import annotations

from typing import Literal, TypedDict

type AllRoles = Literal["system", "user", "assistant"]
type UserAssistantRoles = Literal["user", "assistant"]


class ConversationTurn[RoleT: AllRoles = AllRoles](TypedDict):
    role: RoleT
    content: str


type _Conversation[RoleT: AllRoles] = list[ConversationTurn[RoleT]]
type Conversation = _Conversation[AllRoles]