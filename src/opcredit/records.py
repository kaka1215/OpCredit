from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    ENTITY = "entity_and_direction_selection"
    RELATION = "relation_selection"
    ANSWER = "answer_emission"


class Direction(str, Enum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"


@dataclass(frozen=True)
class Context:
    visited_entities: frozenset[str]
    remaining_budget: int
    phase: Phase


@dataclass(frozen=True)
class NavigationState:
    active_entity: str
    visited_entities: frozenset[str]
    completed_hops: int
    max_hops: int

    def __post_init__(self) -> None:
        if not isinstance(self.active_entity, str) or not self.active_entity:
            raise ValueError("An active entity identifier is required.")
        if isinstance(self.visited_entities, str):
            raise TypeError("Pass a collection of visited entity identifiers.")
        visited = frozenset(self.visited_entities)
        if any(not isinstance(entity, str) or not entity for entity in visited):
            raise ValueError("Visited entities must have nonempty identifiers.")
        if self.active_entity not in visited:
            raise ValueError("The active entity must occur in the visited entity set.")
        if type(self.completed_hops) is not int or type(self.max_hops) is not int:
            raise TypeError("Hop counts must be integers.")
        if self.max_hops < 1 or not 0 <= self.completed_hops <= self.max_hops:
            raise ValueError("Hop counts must satisfy 0 <= completed_hops <= max_hops.")
        object.__setattr__(self, "visited_entities", visited)

    def context(self, phase: Phase) -> Context:
        return Context(self.visited_entities, self.max_hops - self.completed_hops, phase)


@dataclass(frozen=True)
class Operation:
    phase: Phase
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase):
            raise TypeError("Use a Phase value for the operation phase.")
        if isinstance(self.arguments, str):
            raise TypeError("Pass operation arguments as a tuple.")
        arguments = tuple(self.arguments)
        if any(not isinstance(value, str) or not value for value in arguments):
            raise ValueError("Operation arguments must be nonempty strings.")
        if self.phase is Phase.ENTITY:
            if len(arguments) != 2 or arguments[1] not in {d.value for d in Direction}:
                raise ValueError("Entity selection requires an entity and a direction.")
        elif self.phase is Phase.RELATION:
            if len(arguments) != 1:
                raise ValueError("Relation selection requires one relation identifier.")
        else:
            arguments = tuple(sorted(set(arguments)))
        object.__setattr__(self, "arguments", arguments)

    @classmethod
    def entity(cls, entity: str, direction: Direction) -> Operation:
        return cls(Phase.ENTITY, (entity, Direction(direction).value))

    @classmethod
    def relation(cls, relation: str) -> Operation:
        return cls(Phase.RELATION, (relation,))

    @classmethod
    def answer(cls, entities: tuple[str, ...]) -> Operation:
        return cls(Phase.ANSWER, entities)


Frontier = tuple[str | None, ...] | None


@dataclass(frozen=True)
class Decision:
    state: NavigationState
    operation: Operation
    token_start: int
    token_end: int
    frontier_after: Frontier

    def __post_init__(self) -> None:
        if type(self.token_start) is not int or type(self.token_end) is not int:
            raise TypeError("Token positions must be integers.")
        if not 0 <= self.token_start < self.token_end:
            raise ValueError("Each decision must have a nonempty token span.")
        if self.frontier_after is not None:
            if isinstance(self.frontier_after, str):
                raise TypeError("Pass a frontier as a collection of entity identifiers.")
            frontier = tuple(self.frontier_after)
            if any(e is not None and (not isinstance(e, str) or not e) for e in frontier):
                raise ValueError("A frontier entity must be a nonempty identifier or None.")
            if self.operation.phase is Phase.ENTITY and len(frontier) != 1:
                raise ValueError("Entity selection must return a singleton frontier.")
            if self.operation.phase is Phase.ENTITY and frontier[0] is not None:
                if frontier[0] != self.operation.arguments[0]:
                    raise ValueError("The entity frontier must contain the selected entity.")
            if self.operation.phase is Phase.ANSWER and all(e is not None for e in frontier):
                if set(frontier) != set(self.operation.arguments):
                    raise ValueError("The answer frontier must contain the predicted answer set.")
            object.__setattr__(self, "frontier_after", frontier)

    @property
    def context(self) -> Context:
        return self.state.context(self.operation.phase)

    @property
    def frontier_before(self) -> tuple[str]:
        return (self.state.active_entity,)


@dataclass(frozen=True)
class Rollout:
    question_id: str
    rollout_id: str
    terminal_reward: float
    decisions: tuple[Decision, ...]
    num_generated_tokens: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value for value in (self.question_id, self.rollout_id)
        ):
            raise ValueError("Question and rollout identifiers are required.")
        if not math.isfinite(self.terminal_reward):
            raise ValueError("Terminal rewards must be finite.")
        if type(self.num_generated_tokens) is not int or self.num_generated_tokens < 1:
            raise ValueError("A rollout must contain generated tokens.")
        decisions = tuple(self.decisions)
        cursor = 0
        for index, decision in enumerate(decisions):
            if decision.token_start != cursor:
                raise ValueError("Decision spans must partition the generated tokens.")
            if decision.operation.phase is Phase.ANSWER and index != len(decisions) - 1:
                raise ValueError("Answer emission must be the last decision.")
            cursor = decision.token_end
        if cursor != self.num_generated_tokens:
            raise ValueError("Decision spans must cover all generated tokens.")
        object.__setattr__(self, "decisions", decisions)


class DecisionRecorder:
    """Record reasoning and operation tokens together, excluding observations."""

    def __init__(self) -> None:
        self._decisions: list[Decision] = []
        self._num_tokens = 0

    def record(
        self,
        state: NavigationState,
        operation: Operation,
        num_generated_tokens: int,
        frontier_after: Frontier,
    ) -> Decision:
        if self._decisions and self._decisions[-1].operation.phase is Phase.ANSWER:
            raise ValueError("Cannot record a decision after answer emission.")
        if type(num_generated_tokens) is not int or num_generated_tokens < 1:
            raise ValueError("Record the full nonempty generated decision span.")
        decision = Decision(
            state,
            operation,
            self._num_tokens,
            self._num_tokens + num_generated_tokens,
            frontier_after,
        )
        self._decisions.append(decision)
        self._num_tokens = decision.token_end
        return decision

    def finish(self, question_id: str, rollout_id: str, terminal_reward: float) -> Rollout:
        return Rollout(
            question_id,
            rollout_id,
            terminal_reward,
            tuple(self._decisions),
            self._num_tokens,
        )
