from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

from opcredit.records import Context, Operation, Rollout


@dataclass(frozen=True)
class OutcomeEstimate:
    operation_return: float
    context_return: float
    reward_std: float
    credit: float


def _group_rollouts(rollouts: Sequence[Rollout]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    identifiers: set[tuple[str, str]] = set()
    for index, rollout in enumerate(rollouts):
        identifier = (rollout.question_id, rollout.rollout_id)
        if identifier in identifiers:
            raise ValueError("Rollout identifiers must be unique within each question.")
        identifiers.add(identifier)
        groups[rollout.question_id].append(index)
    return dict(groups)


def _reward_scale(rollouts: Sequence[Rollout], indices: list[int]) -> float:
    rewards = [rollouts[i].terminal_reward for i in indices]
    return stdev(rewards) if len(rewards) > 1 else 0.0


def _check_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")


def branch_outcome_credit(
    rollouts: Sequence[Rollout], epsilon: float
) -> tuple[tuple[OutcomeEstimate, ...], ...]:
    """Compute BOC from complete same-question rollout groups."""
    _check_epsilon(epsilon)
    groups = _group_rollouts(rollouts)
    output: list[tuple[OutcomeEstimate, ...]] = [()] * len(rollouts)

    for indices in groups.values():
        contexts: dict[tuple[str, Context], set[int]] = defaultdict(set)
        branches: dict[tuple[str, Context, Operation], set[int]] = defaultdict(set)
        mean = fmean(rollouts[i].terminal_reward for i in indices)
        scale = _reward_scale(rollouts, indices)
        for i in indices:
            for decision in rollouts[i].decisions:
                key = (decision.state.active_entity, decision.context)
                contexts[key].add(i)
                branches[(*key, decision.operation)].add(i)

        values = {
            key: fmean(rollouts[i].terminal_reward for i in sorted(support))
            for key, support in contexts.items()
        }
        estimates: dict[tuple[str, Context, Operation], tuple[float, float]] = {}
        for key, support in branches.items():
            q_value = fmean(rollouts[i].terminal_reward for i in sorted(support))
            v_value = values[key[:2]]
            estimates[key] = (q_value, v_value)
        for i in indices:
            assigned = []
            for decision in rollouts[i].decisions:
                key = (decision.state.active_entity, decision.context, decision.operation)
                q_value, v_value = estimates[key]
                contrast = q_value - v_value
                numerator = contrast if contrast != 0.0 else rollouts[i].terminal_reward - mean
                assigned.append(
                    OutcomeEstimate(q_value, v_value, scale, numerator / (scale + epsilon))
                )
            output[i] = tuple(assigned)
    return tuple(output)


def trajectory_credit(rollouts: Sequence[Rollout], epsilon: float) -> tuple[float, ...]:
    """Compute question-normalized trajectory advantages."""
    _check_epsilon(epsilon)
    output = [0.0] * len(rollouts)
    for indices in _group_rollouts(rollouts).values():
        mean = fmean(rollouts[i].terminal_reward for i in indices)
        scale = _reward_scale(rollouts, indices)
        for i in indices:
            output[i] = (rollouts[i].terminal_reward - mean) / (scale + epsilon)
    return tuple(output)
