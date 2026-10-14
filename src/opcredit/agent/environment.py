from __future__ import annotations

import json
import re
from dataclasses import dataclass

from opcredit.agent.kg import visible_neighbors
from opcredit.agent.prompts import (
    entity_observation,
    initial_observation,
    operation_text,
    relation_observation,
)
from opcredit.records import Direction, Frontier, NavigationState, Operation


def normalize_name(name: str) -> str:
    return " ".join(name.casefold().strip().split())


@dataclass(frozen=True)
class EnvironmentConfig:
    max_hops: int
    relation_candidates: int
    entity_candidates: int
    neighbor_limit: int
    mediator_predicate_limit: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("Environment limits must be positive integers.")


class InvalidOperation(ValueError):
    pass


class NavigationEnvironment:
    def __init__(self, graph, ranker, question: str, topic: str, config: EnvironmentConfig):
        self.graph = graph
        self.ranker = ranker
        self.config = config
        self.question = question
        self.active = topic
        self.visited = {topic}
        self.hops = 0
        self.direction = Direction.OUTGOING
        self.mode = "entity"
        self.relations: list[str] = []
        self.entities = {topic: graph.name(topic)}
        if not self.entities[topic]:
            raise ValueError("The topic entity must have a display name.")
        self.path = [self.entities[topic]]
        self.explored_paths: list[str] = []
        self.predicted_names: list[str] = []
        self.done = False

    @property
    def state(self) -> NavigationState:
        return NavigationState(
            self.active, frozenset(self.visited), self.hops, self.config.max_hops
        )

    @property
    def initial_prompt(self) -> str:
        return initial_observation(self.question, self.graph.name(self.active))

    def _choose(self, keys: list[str], texts: list[str], limit: int, required=()) -> list[str]:
        order = self.ranker.order(self.question, texts)
        if sorted(order) != list(range(len(keys))):
            raise ValueError("The ranker must return a permutation of candidate indices.")
        required = list(dict.fromkeys(required))
        if not set(required).issubset(keys) or len(required) > limit:
            raise ValueError("The supervised operation is outside the retrieved candidates.")
        selected = [keys[i] for i in order[:limit]]
        for key in required:
            if key not in selected:
                if len(selected) >= limit:
                    removable = next(
                        i for i in reversed(range(len(selected))) if selected[i] not in required
                    )
                    selected.pop(removable)
                selected.append(key)
        return selected

    def resolve(self, name: str) -> str | None:
        if name in self.entities:
            return name
        matches = [
            key
            for key, label in self.entities.items()
            if normalize_name(label) == normalize_name(name)
        ]
        return matches[0] if len(matches) == 1 else None

    def explore(self, entity: str, direction: Direction, required_relation: str | None = None):
        if (
            self.mode != "entity"
            or self.hops >= self.config.max_hops
            or entity not in self.entities
        ):
            raise InvalidOperation("Select a visible entity while navigation budget remains.")
        self.active = entity
        self.visited.add(entity)
        self.direction = Direction(direction)
        if self.hops:
            self.path.append(self.graph.name(entity))
        candidates = list(dict.fromkeys(self.graph.predicates(entity, self.direction)))
        context = " ".join(self.path) + " " if self.hops else ""
        self.relations = self._choose(
            candidates,
            [context + p for p in candidates],
            self.config.relation_candidates,
            (required_relation,) if required_relation else (),
        )
        self.mode = "relation" if self.relations else "answer"
        observation = (
            relation_observation(self.graph.name(entity), self.direction.value, self.relations)
            if self.relations
            else self.answer_prompt()
        )
        return observation, (entity,)

    def follow(self, relation: str, required_entities=()):
        if self.mode != "relation" or relation not in self.relations:
            raise InvalidOperation("Select a relation from the current observation.")
        candidates = visible_neighbors(
            self.graph,
            self.active,
            relation,
            self.direction,
            self.config.neighbor_limit,
            self.config.mediator_predicate_limit,
        )
        candidates = {key: name for key, name in candidates.items() if key not in self.visited}
        keys = sorted(candidates)
        context = " ".join(self.path)
        selected = self._choose(
            keys,
            [f"{context} {relation} {candidates[key]}" for key in keys],
            self.config.entity_candidates,
            required_entities,
        )
        self.entities = {key: candidates[key] for key in selected}
        self.path.append(f"-> [{relation}]")
        self.explored_paths.append(f"{self.graph.name(self.active)} --{relation}-->")
        self.hops += 1
        self.mode = "answer" if self.hops >= self.config.max_hops or not selected else "entity"
        return self.answer_prompt(), tuple(selected)

    def answer_prompt(self) -> str:
        at_max_depth = self.hops >= self.config.max_hops
        paths = self.explored_paths if at_max_depth else self.explored_paths[-1:]
        return entity_observation(
            self.question,
            list(self.entities.values()),
            paths or [self.graph.name(self.active)],
            at_max_depth,
        )

    def execute(self, text: str) -> tuple[Operation, Frontier, str]:
        content = operation_text(text)
        answer = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        if answer:
            try:
                names = json.loads(answer.group(1))
            except json.JSONDecodeError as error:
                raise InvalidOperation("Answer with a valid JSON list of names.") from error
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                raise InvalidOperation("Answer with a JSON list of names.")
            names = list(dict.fromkeys(name.strip() for name in names if name.strip()))
            mapped = tuple(self.resolve(name) for name in names)
            keys = tuple(
                mid or f"unresolved:{normalize_name(name)}"
                for name, mid in zip(names, mapped, strict=True)
            )
            self.predicted_names = names
            self.done = True
            return Operation.answer(keys), mapped, ""
        explore = re.fullmatch(
            r'<explore\s+direction="(outgoing|incoming)">\s*(.*?)\s*</explore>',
            content,
            re.DOTALL,
        )
        if explore:
            entity = self.resolve(explore.group(2))
            if entity is None:
                raise InvalidOperation(
                    "Select one unambiguous entity from the current observation."
                )
            direction = Direction(explore.group(1))
            observation, frontier = self.explore(entity, direction)
            return Operation.entity(entity, direction), frontier, observation
        observation, frontier = self.follow(content)
        return Operation.relation(content), frontier, observation

    def invalid_operation(self, text: str) -> Operation:
        key = "unresolved:" + operation_text(text)
        if self.mode == "relation":
            return Operation.relation(key)
        return Operation.entity(key, self.direction)
