import math
import re
import string


def terminal_task_reward(hit_at_one: float, f1: float) -> float:
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (hit_at_one, f1)):
        raise ValueError("Hit@1 and F1 must lie in [0, 1].")
    return 0.5 * hit_at_one + 0.5 * f1


def normalized_answer(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def task_reward(predicted: list[str], references: tuple[str, ...]) -> float:
    prediction = {normalized_answer(name) for name in predicted} - {""}
    gold = {normalized_answer(name) for name in references} - {""}
    if not gold:
        raise ValueError("Training questions must have nonempty reference answers.")
    matches = len(prediction & gold)
    return terminal_task_reward(float(matches > 0), 2 * matches / (len(prediction) + len(gold)))
