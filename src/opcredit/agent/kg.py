from __future__ import annotations

import json
import re
from functools import lru_cache
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from opcredit.credit.ftc import is_task_predicate
from opcredit.records import Direction

NAMESPACE = "http://rdf.freebase.com/ns/"


class FreebaseGraph:
    def __init__(self, endpoint: str, timeout: float):
        if urlparse(endpoint).scheme not in {"http", "https"}:
            raise ValueError("Supply an HTTP(S) SPARQL endpoint.")
        self.endpoint = endpoint
        self.timeout = timeout
        self._query = lru_cache(maxsize=8192)(self._query)
        self.name = lru_cache(maxsize=32768)(self.name)
        self.predicates = lru_cache(maxsize=32768)(self.predicates)
        self.neighbors = lru_cache(maxsize=65536)(self.neighbors)

    @staticmethod
    def _term(identifier: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", identifier):
            raise ValueError(f"Invalid Freebase identifier: {identifier!r}")
        return f"<{NAMESPACE}{identifier}>"

    def _query(self, query: str) -> tuple[dict, ...]:
        body = urlencode({"query": query, "format": "application/sparql-results+json"}).encode()
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return tuple(result["results"]["bindings"])

    def name(self, entity: str) -> str:
        if entity.startswith("lit:"):
            return entity[4:]
        query = (
            f"SELECT ?label WHERE {{ {self._term(entity)} "
            f"<{NAMESPACE}type.object.name> ?label . "
            "FILTER(lang(?label) = 'en') } LIMIT 1"
        )
        rows = self._query(query)
        return rows[0]["label"]["value"] if rows else ""

    def predicates(self, entity: str, direction: Direction) -> tuple[str, ...]:
        if entity.startswith("lit:"):
            return ()
        anchor = self._term(entity)
        edge = f"{anchor} ?p ?x" if direction == Direction.OUTGOING else f"?x ?p {anchor}"
        rows = self._query(f"SELECT DISTINCT ?p WHERE {{ {edge} . }}")
        return tuple(
            row["p"]["value"].removeprefix(NAMESPACE)
            for row in rows
            if row["p"]["value"].startswith(NAMESPACE) and is_task_predicate(row["p"]["value"])
        )

    def neighbors(
        self, entity: str, predicate: str, direction: Direction, limit: int
    ) -> tuple[str, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("The neighbor limit must be a positive integer.")
        if entity.startswith("lit:"):
            return ()
        anchor, relation = self._term(entity), self._term(predicate)
        edge = (
            f"{anchor} {relation} ?x"
            if direction == Direction.OUTGOING
            else f"?x {relation} {anchor}"
        )
        rows = self._query(f"SELECT DISTINCT ?x WHERE {{ {edge} . }} LIMIT {limit}")
        values = []
        for row in rows:
            item = row["x"]
            if item["type"] == "uri" and item["value"].startswith(NAMESPACE):
                values.append(item["value"].removeprefix(NAMESPACE))
            elif item["type"] in {"literal", "typed-literal"}:
                values.append("lit:" + item["value"])
        return tuple(dict.fromkeys(values))


def visible_neighbors(
    graph,
    entity: str,
    relation: str,
    direction: Direction,
    limit: int,
    mediator_predicate_limit: int,
):
    found: dict[str, str] = {}
    for candidate in graph.neighbors(entity, relation, direction, limit):
        name = graph.name(candidate)
        if name:
            found[candidate] = name
            continue
        predicates = dict.fromkeys(
            (
                *graph.predicates(candidate, Direction.OUTGOING),
                *graph.predicates(candidate, Direction.INCOMING),
            )
        )
        for predicate in list(predicates)[:mediator_predicate_limit]:
            for child_direction in Direction:
                for child in graph.neighbors(candidate, predicate, child_direction, limit):
                    child_name = graph.name(child)
                    if child != entity and child_name:
                        found[child] = child_name
    return found


class SemanticRanker:
    def __init__(self, model: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model, device=device, trust_remote_code=False)

    def order(self, question: str, texts: list[str]) -> list[int]:
        if not texts:
            return []
        vectors = self.encoder.encode(
            [question, *texts], normalize_embeddings=True, show_progress_bar=False
        )
        scores = vectors[1:] @ vectors[0]
        return sorted(range(len(texts)), key=lambda index: (-float(scores[index]), index))
