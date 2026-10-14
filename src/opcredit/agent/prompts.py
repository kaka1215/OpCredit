import json
import re

SYSTEM_PROMPT = (
    "You are a KGQA agent that answers questions by navigating a knowledge graph.\n"
    "Explore the graph step by step:\n"
    "\n"
    "1. Choose an entity and a direction to explore:\n"
    '   <explore direction="outgoing">entity_name</explore> — follow outgoing edges from entity\n'
    '   <explore direction="incoming">entity_name</explore> — follow incoming edges to entity\n'
    "   Reason in <think > tags before each action.\n"
    "\n"
    "2. After seeing the available relations, select the most relevant one by outputting its name exactly.\n"
    "\n"
    "3. After seeing the resulting entities, either:\n"
    '   - Answer: <answer>["name1", "name2"]</answer> — JSON list of entity names exactly as shown\n'
    '   - Continue: <explore direction="outgoing/incoming">entity_name</explore>\n'
    "\n"
    "Important:\n"
    "- Check ALL entities before answering. Do not pick the first plausible one.\n"
    "- If multiple entities satisfy the question, list them all.\n"
    "- Do NOT fabricate facts. Only use information visible in names."
)


def initial_observation(question: str, entity: str) -> str:
    return (
        f"Question: {question}\nCurrent entity: {entity}\n"
        "Explore outgoing or incoming relations from this entity."
    )


def relation_observation(entity: str, direction: str, relations: list[str]) -> str:
    lines = "\n".join(f"  {relation}" for relation in relations)
    preposition = "from" if direction == "outgoing" else "to"
    return (
        f"{direction.title()} relations ({preposition} {entity}):\n{lines}\n\n"
        "Select the most relevant relation name exactly as shown."
    )


def entity_observation(question: str, entities: list[str], path: list[str], final: bool) -> str:
    names = "\n".join(f"  {entity}" for entity in entities)
    history = "\n  ".join(path)
    instruction = (
        "You have reached the maximum exploration depth. Provide your final answer in <answer>\n"
        "tags as a JSON list of entity names copied exactly from the results above."
        if final
        else "Based on the entities above, answer the question or continue exploring. If answering,\n"
        "your answer MUST be one of the Entities listed above, copied exactly as shown."
    )
    return (
        f"<information>\nQuestion: {question}\nPaths explored:\n  {history}\n"
        f"Entities:\n{names}\n</information>\n\n{instruction}"
    )


def reasoning(text: str, operation: str) -> str:
    return f"<think>\n{text}\n</think>\n{operation}"


def explore_text(entity: str, direction: str) -> str:
    return reasoning(
        f'I need to inspect {direction} relations for "{entity}".',
        f'<explore direction="{direction}">{entity}</explore>',
    )


def relation_text(entity: str, relation: str) -> str:
    return reasoning(
        f'The relation "{relation}" is relevant, so I will follow it from "{entity}".',
        relation,
    )


def answer_text(names: list[str]) -> str:
    return reasoning(
        "The latest entity observation contains the answer names, so I can answer directly.",
        f"<answer>{json.dumps(names, ensure_ascii=False)}</answer>",
    )


def operation_text(text: str) -> str:
    return re.sub(r"<think\s*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()
