from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

from opcredit.agent.environment import (
    EnvironmentConfig,
    NavigationEnvironment,
    normalize_name,
)
from opcredit.agent.kg import FreebaseGraph, SemanticRanker
from opcredit.agent.modeling import configured_arguments
from opcredit.agent.prompts import (
    SYSTEM_PROMPT,
    answer_text,
    explore_text,
    relation_text,
)
from opcredit.records import Direction


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        data = json.load(stream)
    if isinstance(data, dict):
        data = data.get("results", data.get("Questions", data))
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError("Expected a JSON array, results/Questions array, or JSONL records.")
    return data


def record_keys(row: dict, dataset: str) -> set[str]:
    identifier = row.get("question_id", row.get("QuestionId", row.get("ID", row.get("qid", ""))))
    question = row.get("question", row.get("ProcessedQuestion", row.get("RawQuestion", "")))
    keys = {f"id:{dataset}:{identifier}"} if identifier != "" else set()
    if question:
        keys.add("question:" + normalize_name(question))
    source = row.get("webqsp_ID", row.get("webqsp_id", row.get("source_question_id")))
    if dataset == "webqsp":
        source = identifier
    if source is not None and source != "":
        keys.add("source:webqsp:" + str(source))
    return keys


def query_graph_path(graph: dict) -> tuple[str, tuple[str, ...]]:
    nodes = {node["nid"]: node for node in graph.get("nodes", [])}
    topics = [node for node in nodes.values() if node.get("node_type") == "entity"]
    if not topics:
        return "", ()
    start = topics[0]["nid"]
    adjacency: dict = {}
    for edge in graph.get("edges", []):
        adjacency.setdefault(edge["start"], []).append((edge["end"], edge["relation"]))
        adjacency.setdefault(edge["end"], []).append((edge["start"], edge["relation"]))
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current != start and nodes[current].get("question_node") == 1:
            return topics[0]["id"], path
        for target, relation in adjacency.get(current, []):
            if target not in seen:
                seen.add(target)
                queue.append((target, (*path, relation)))
    return topics[0]["id"], ()


@dataclass(frozen=True)
class Question:
    question_id: str
    dataset: str
    question: str
    topic_entity: str
    answer_entities: tuple[str, ...]
    answer_names: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def from_dict(cls, row: dict) -> Question:
        return cls(
            row["question_id"],
            row["dataset"],
            row["question"],
            row["topic_entity"],
            tuple(row["answer_entities"]),
            tuple(row["answer_names"]),
            tuple(tuple(path) for path in row.get("paths", [])),
        )


def normalize_record(row: dict, dataset: str, graph, max_hops: int) -> Question | None:
    identifier = row.get("question_id", row.get("QuestionId", row.get("ID", row.get("qid", ""))))
    question = row.get("question", row.get("ProcessedQuestion", row.get("RawQuestion", "")))
    topic = row.get("topic_entity", "")
    answers = list(row.get("answer_entities", []))
    names = list(row.get("answer_names", []))
    supplied_names = dict(zip(answers, names, strict=True)) if len(answers) == len(names) else {}
    paths: list[tuple[str, ...]] = []
    if "Parses" in row:
        parses = row["Parses"]
        if not parses:
            return None
        topic = parses[0].get("TopicEntityMid", "")
        for parse in parses:
            if parse.get("TopicEntityMid") != topic:
                continue
            path = parse.get("InferentialChain") or []
            if path:
                paths.append(tuple(path))
            for answer in parse.get("Answers", []):
                value = answer.get("AnswerArgument", "")
                if value:
                    entity = value if value.startswith(("m.", "g.")) else "lit:" + value
                    answers.append(entity)
                    if answer.get("EntityName"):
                        supplied_names.setdefault(entity, answer["EntityName"])
    elif "graph_query" in row:
        topic, path = query_graph_path(row["graph_query"])
        if path:
            paths.append(path)
        for answer in row.get("answer", []):
            value = answer.get("answer_argument", "")
            if value is None or value == "":
                continue
            if answer.get("answer_type") == "Entity":
                entity = value
            elif answer.get("answer_type") == "Value":
                entity = "lit:" + str(value)
            else:
                continue
            answers.append(entity)
            if answer.get("entity_name"):
                supplied_names.setdefault(entity, answer["entity_name"])
    else:
        gold = tuple(relation for relation in row.get("gold_relations", []) if relation)
        candidates = sorted(row.get("oracle_paths", []), key=lambda p: -p.get("sim_score", 0.0))
        target_hops = len(gold) or min((len(p.get("relations", [])) for p in candidates), default=0)
        seen_prefixes: set[str] = set()
        for candidate in candidates:
            relations = tuple(candidate.get("relations", []))
            if len(relations) != target_hops or not relations:
                continue
            if target_hops > 2 and relations[0] in seen_prefixes:
                continue
            paths.append(relations)
            seen_prefixes.add(relations[0])
            if len(paths) >= (1 if target_hops <= 2 else 3):
                break
        if not paths and gold:
            paths.append(gold)
    if not question or not topic or identifier == "" or not graph.name(topic):
        return None
    answers = list(dict.fromkeys(answers))
    names = [graph.name(entity) or supplied_names.get(entity, "") for entity in answers]
    if not answers or not all(names):
        return None
    names = list(dict.fromkeys(names))
    paths = list(dict.fromkeys(path for path in paths if 0 < len(path) <= max_hops))
    return Question(
        f"{dataset}:{identifier}",
        dataset,
        question,
        topic,
        tuple(answers),
        tuple(names),
        tuple(paths),
    )


def ground_path(
    graph, topic: str, relations: tuple[str, ...], answers: tuple[str, ...], neighbor_limit: int
):
    layers = [{topic}]
    edges = []
    for relation in relations:
        current = {}
        for parent in sorted(layers[-1]):
            links = []
            for direction in Direction:
                links.extend(
                    (child, direction)
                    for child in graph.neighbors(parent, relation, direction, neighbor_limit)
                )
            current[parent] = tuple(dict.fromkeys(links))
        edges.append(current)
        layers.append({child for links in current.values() for child, _ in links})
        if not layers[-1]:
            return None
    valid = [set() for _ in layers]
    valid[-1] = layers[-1] & set(answers)
    for index in reversed(range(len(relations))):
        valid[index] = {
            parent
            for parent, links in edges[index].items()
            if any(child in valid[index + 1] for child, _ in links)
        }
    if topic not in valid[0]:
        return None
    steps = []
    current = topic
    for index, relation in enumerate(relations):
        options = sorted(
            (child, direction)
            for child, direction in edges[index][current]
            if child in valid[index + 1]
        )
        child, direction = options[0]
        steps.append((current, relation, direction, child))
        current = child
    return steps


def build_trajectory(
    question: Question,
    relations: tuple[str, ...],
    graph,
    ranker,
    environment_config: EnvironmentConfig,
    grounding_neighbor_limit: int,
) -> dict | None:
    steps = ground_path(
        graph, question.topic_entity, relations, question.answer_entities, grounding_neighbor_limit
    )
    if not steps:
        return None
    env = NavigationEnvironment(
        graph, ranker, question.question, question.topic_entity, environment_config
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": env.initial_prompt},
    ]
    index = 0
    while index < len(steps):
        entity, relation, direction, child = steps[index]
        if not graph.name(entity):
            return None
        next_index = index + 1
        if not graph.name(child) and next_index < len(steps):
            child = steps[next_index][3]
            next_index += 1
        final = next_index == len(steps)
        try:
            observation, _ = env.explore(entity, direction, required_relation=relation)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": explore_text(graph.name(entity), direction.value),
                    },
                    {"role": "user", "content": observation},
                ]
            )
            observation, _ = env.follow(relation, required_entities=(child,))
        except ValueError:
            return None
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": relation_text(graph.name(entity), relation),
                },
                {"role": "user", "content": observation},
            ]
        )
        if final:
            selected = sorted(set(env.entities) & set(question.answer_entities))
            if not selected:
                return None
            messages.append(
                {
                    "role": "assistant",
                    "content": answer_text([env.entities[mid] for mid in selected]),
                }
            )
        index = next_index
    return {"question_id": question.question_id, "messages": messages}


def parse_source(value: str) -> tuple[str, str]:
    dataset, separator, path = value.partition("=")
    if not separator or dataset not in {"cwq", "webqsp", "grailqa"} or not path:
        raise argparse.ArgumentTypeError("Use dataset=path with cwq, webqsp, or grailqa.")
    return dataset, path


def prepare(
    sources,
    exclusions,
    output: Path,
    graph,
    ranker,
    max_hops: int,
    environment_config: EnvironmentConfig,
    grounding_neighbor_limit: int,
):
    datasets = {dataset for dataset, _ in sources}
    if "grailqa" in datasets and len(datasets) > 1:
        raise ValueError("Prepare GrailQA separately from the CWQ/WebQSP pool.")
    excluded = set()
    for dataset, path in exclusions:
        for row in read_records(path):
            excluded.update(record_keys(row, dataset))
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("sft.jsonl", "rl.jsonl", "preparation.json")):
        raise FileExistsError("Use a new output directory to avoid overwriting prepared data.")
    counts = Counter()
    seen = set()
    with (
        (output / "sft.jsonl").open("w", encoding="utf-8") as sft,
        (output / "rl.jsonl").open("w", encoding="utf-8") as rl,
    ):
        for dataset, path in sources:
            for row in read_records(path):
                counts["input_questions"] += 1
                keys = record_keys(row, dataset)
                if keys & excluded:
                    counts["excluded_evaluation_source"] += 1
                    continue
                identity = {key for key in keys if not key.startswith("source:")}
                if identity & seen:
                    counts["duplicate_question"] += 1
                    continue
                question = normalize_record(row, dataset, graph, max_hops)
                if question is None:
                    counts["incomplete_question"] += 1
                    continue
                seen.update(identity)
                rl_row = asdict(question)
                rl_row.pop("paths")
                rl.write(json.dumps(rl_row, ensure_ascii=False) + "\n")
                counts["rl_questions"] += 1
                for path_relations in question.paths:
                    trajectory = build_trajectory(
                        question,
                        path_relations,
                        graph,
                        ranker,
                        environment_config,
                        grounding_neighbor_limit,
                    )
                    if trajectory is None:
                        counts["ungrounded_or_unavailable_path"] += 1
                        continue
                    sft.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                    counts["sft_trajectories"] += 1
    (output / "preparation.json").write_text(json.dumps(dict(counts), indent=2) + "\n")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare template-based SFT traces and RL questions."
    )
    parser.add_argument("--input", action="append", required=True, type=parse_source)
    parser.add_argument("--exclude", action="append", required=True, type=parse_source)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", required=True)
    parser.add_argument("--query-timeout", type=float, required=True)
    for field in (
        "max-gold-hops",
        "grounding-neighbor-limit",
        "max-hops",
        "relation-candidates",
        "entity-candidates",
        "neighbor-limit",
        "mediator-predicate-limit",
    ):
        parser.add_argument("--" + field, type=int, required=True)
    args = configured_arguments(parser)
    if args.max_gold_hops < 1:
        parser.error("--max-gold-hops must be positive")
    if args.grounding_neighbor_limit < 1 or args.query_timeout <= 0:
        parser.error("Grounding limits and query timeout must be positive.")
    config = EnvironmentConfig(
        args.max_hops,
        args.relation_candidates,
        args.entity_candidates,
        args.neighbor_limit,
        args.mediator_predicate_limit,
    )
    graph = FreebaseGraph(args.endpoint, args.query_timeout)
    ranker = SemanticRanker(args.embedding_model, args.embedding_device)
    print(
        json.dumps(
            prepare(
                args.input,
                args.exclude,
                args.output,
                graph,
                ranker,
                args.max_gold_hops,
                config,
                args.grounding_neighbor_limit,
            )
        )
    )


if __name__ == "__main__":
    main()
