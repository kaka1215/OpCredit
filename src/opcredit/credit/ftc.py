from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain, islice
from typing import Protocol

from opcredit.records import Decision, Direction, Frontier


class KnowledgeGraph(Protocol):
    def predicates(self, entity: str, direction: Direction) -> Iterable[str]: ...

    def neighbors(
        self, entity: str, predicate: str, direction: Direction, limit: int
    ) -> Iterable[str]: ...


def is_task_predicate(predicate: str) -> bool:
    predicate = predicate.removeprefix("http://rdf.freebase.com/ns/")
    return (
        bool(predicate)
        and "sameAs" not in predicate
        and not predicate.startswith(
            (
                "http://www.w3.org/",
                "http://rdf.freebase.com/key/",
                "type.",
                "common.",
                "base.",
                "freebase.",
                "kg.",
                "user.",
                "dataworld.",
            )
        )
    )


@dataclass(frozen=True)
class AnswerNeighborhood:
    gold_entities: frozenset[str]
    one_hop_entities: frozenset[str]

    def __post_init__(self) -> None:
        if isinstance(self.gold_entities, str) or isinstance(self.one_hop_entities, str):
            raise TypeError("Pass collections of mapped entity identifiers.")
        gold = frozenset(self.gold_entities)
        neighbors = frozenset(self.one_hop_entities)
        if any(not isinstance(e, str) or not e for e in gold | neighbors):
            raise ValueError("Neighborhoods must contain mapped entity identifiers.")
        object.__setattr__(self, "gold_entities", gold)
        object.__setattr__(self, "one_hop_entities", neighbors)

    @classmethod
    def from_graph(
        cls,
        graph: KnowledgeGraph,
        gold_entities: Iterable[str],
        max_predicates: int,
        max_neighbors: int,
    ) -> AnswerNeighborhood:
        if type(max_predicates) is not int or type(max_neighbors) is not int:
            raise TypeError("Retrieval limits must be integers.")
        if max_predicates < 1 or max_neighbors < 1:
            raise ValueError("Retrieval limits must be positive.")
        if isinstance(gold_entities, str):
            raise TypeError("Pass a collection of gold entity identifiers.")
        gold = frozenset(gold_entities)
        cls(gold, frozenset())
        neighbors: set[str] = set()
        for entity in sorted(gold):
            predicates = chain(
                graph.predicates(entity, Direction.OUTGOING),
                graph.predicates(entity, Direction.INCOMING),
            )
            selected = islice(dict.fromkeys(filter(is_task_predicate, predicates)), max_predicates)
            for predicate in selected:
                for direction in Direction:
                    returned = graph.neighbors(entity, predicate, direction, max_neighbors)
                    neighbors.update(islice(returned, max_neighbors))
        return cls(gold, frozenset(neighbors))

    def proximity(self, entity: str) -> int:
        if entity in self.gold_entities:
            return 0
        return 1 if entity in self.one_hop_entities else 2

    def potential(self, frontier: Frontier) -> float | None:
        if isinstance(frontier, str):
            raise TypeError("Pass a frontier as a collection of entity identifiers.")
        if not self.gold_entities or not frontier or any(entity is None for entity in frontier):
            return None
        return float(max(2 - self.proximity(entity) for entity in frontier if entity is not None))

    def transition(self, before: Frontier, after: Frontier) -> float:
        phi_before, phi_after = self.potential(before), self.potential(after)
        if phi_before is None or phi_after is None:
            return 0.0
        return phi_after - phi_before

    def credit(self, decision: Decision) -> float:
        return self.transition(decision.frontier_before, decision.frontier_after)
