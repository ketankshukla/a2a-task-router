"""The wire types, modelled on the Agent2Agent protocol's shape.

Three ideas are borrowed and the rest is deliberately left out:

  AgentCard   a machine-readable advertisement of what an agent can do, served
              at a well-known path. This is how one agent discovers another
              without a human wiring them together.
  Task        a unit of work with an id and a lifecycle, so a caller can ask
              about it later rather than only awaiting a response.
  Message     the envelope, with parts, so a task can carry more than a string.

What is not here: transport, streaming, push notifications, authentication.
This repo is about the routing decision, and those would obscure it.

The one thing worth taking from A2A even if you never adopt it: the capability
declaration is *data*, published by the agent, which means routing can be
learned from it rather than hard-coded against it.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    # A task nothing was confident enough to claim. Distinct from FAILED:
    # nothing went wrong, the router declined to guess.
    UNROUTABLE = "unroutable"


@dataclass(frozen=True)
class Skill:
    """One named capability, with examples.

    The examples are not documentation. They are the training data — see
    routers.LearnedRouter, which fits on exactly these strings. An agent that
    describes its skills well becomes routable without anyone editing the
    router, which is the property that makes a card worth publishing.
    """

    skill_id: str
    name: str
    description: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentCard:
    name: str
    description: str
    url: str
    version: str
    skills: tuple[Skill, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentCard:
        return cls(
            name=raw["name"],
            description=raw["description"],
            url=raw["url"],
            version=raw.get("version", "0.1.0"),
            skills=tuple(
                Skill(
                    skill_id=s["skill_id"],
                    name=s["name"],
                    description=s["description"],
                    examples=tuple(s.get("examples", ())),
                )
                for s in raw.get("skills", ())
            ),
        )


@dataclass(frozen=True)
class Part:
    kind: str
    text: str


@dataclass(frozen=True)
class Message:
    role: str
    parts: tuple[Part, ...]

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.parts if p.kind == "text").strip()

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", parts=(Part("text", text),))

    @classmethod
    def agent(cls, text: str) -> Message:
        return cls(role="agent", parts=(Part("text", text),))


@dataclass
class Task:
    message: Message
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: TaskState = TaskState.SUBMITTED
    assigned_to: str | None = None
    history: list[Message] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "assigned_to": self.assigned_to,
            "message": asdict(self.message),
            "history": [asdict(m) for m in self.history],
            "error": self.error,
        }


@dataclass(frozen=True)
class RoutingDecision:
    """Why a task went where it did.

    `confidence` and `reason` exist so a wrong route is diagnosable after the
    fact. A router that returns only a name gives you nothing to look at when
    it sends a refund question to the code reviewer.
    """

    agent: str | None
    confidence: float
    reason: str

    @property
    def routed(self) -> bool:
        return self.agent is not None
