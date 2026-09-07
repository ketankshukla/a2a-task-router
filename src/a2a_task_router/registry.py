"""Agent discovery: load cards, hand them to a router, dispatch a task.

The orchestrator is deliberately thin. All the judgement lives in the router,
so swapping routing strategies changes one constructor argument and nothing
else — which is what makes the head-to-head comparison in evaluate.py a fair
one rather than a comparison of two different systems.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .protocol import AgentCard, Message, RoutingDecision, Task, TaskState

DATA = Path(__file__).resolve().parents[2] / "data"


class Router(Protocol):
    name: str

    def fit(self, cards: list[AgentCard]): ...

    def route(self, text: str) -> RoutingDecision: ...


def load_cards(path: Path | None = None) -> list[AgentCard]:
    path = path or DATA / "agents.json"
    with path.open(encoding="utf-8") as fh:
        return [AgentCard.from_dict(raw) for raw in json.load(fh)]


@dataclass
class Registry:
    """The set of agents currently advertising themselves."""

    cards: list[AgentCard] = field(default_factory=list)

    def add(self, card: AgentCard) -> Registry:
        self.cards = [c for c in self.cards if c.name != card.name] + [card]
        return self

    def remove(self, name: str) -> Registry:
        self.cards = [c for c in self.cards if c.name != name]
        return self

    def get(self, name: str) -> AgentCard | None:
        return next((c for c in self.cards if c.name == name), None)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cards]


# A handler stands in for the remote agent. In a real deployment this is an
# HTTP call to the url on the card; here it is a function, so the orchestrator
# can be tested without a network.
Handler = Callable[[Task], str]


@dataclass
class Orchestrator:
    registry: Registry
    router: Router
    handlers: dict[str, Handler] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self) -> Orchestrator:
        self.router.fit(self.registry.cards)
        self._fitted = True
        return self

    def submit(self, text: str) -> Task:
        """Route one task and run it.

        Three outcomes, and all three are ordinary rather than exceptional:
        completed, unroutable (nothing was confident enough to claim it), and
        failed (an agent was chosen and raised).
        """
        if not self._fitted:
            self.fit()

        task = Task(message=Message.user(text))
        decision = self.router.route(text)

        if not decision.routed:
            task.state = TaskState.UNROUTABLE
            task.error = decision.reason
            task.history.append(
                Message.agent(f"No agent claimed this task: {decision.reason}")
            )
            return task

        task.assigned_to = decision.agent
        task.state = TaskState.WORKING

        handler = self.handlers.get(decision.agent)
        if handler is None:
            task.state = TaskState.FAILED
            task.error = f"no handler registered for {decision.agent}"
            return task

        try:
            reply = handler(task)
        except Exception as exc:  # noqa: BLE001 - surfaced on the task, not swallowed
            task.state = TaskState.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            task.history.append(Message.agent(f"Agent failed: {task.error}"))
            return task

        task.state = TaskState.COMPLETED
        task.history.append(Message.agent(reply))
        return task
