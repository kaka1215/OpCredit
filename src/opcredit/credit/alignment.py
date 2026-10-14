from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from opcredit.credit.boc import branch_outcome_credit, trajectory_credit
from opcredit.credit.ftc import AnswerNeighborhood
from opcredit.records import Rollout


@dataclass(frozen=True)
class TokenCredits:
    outcome: torch.Tensor
    transition: torch.Tensor


def validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("The token mask must be a nonempty batch-by-token matrix.")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("The token mask must contain only zero and one.")
    valid = mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Each rollout must contain at least one generated token.")
    return valid


@torch.no_grad()
def build_token_credits(
    rollouts: Sequence[Rollout],
    response_mask: torch.Tensor,
    neighborhoods: Mapping[str, AnswerNeighborhood] | Sequence[AnswerNeighborhood] | None = None,
    *,
    use_boc: bool = True,
    use_ftc: bool = True,
    epsilon: float,
) -> TokenCredits:
    """Scatter generated-token spans into a response tensor containing observations."""
    mask = validate_mask(response_mask)
    if len(rollouts) != mask.shape[0]:
        raise ValueError("The number of rollouts must match the response batch.")
    if use_ftc and neighborhoods is None:
        raise ValueError("Provide training-time neighborhoods when FTC is enabled.")
    if neighborhoods is not None and not isinstance(neighborhoods, Mapping):
        if not isinstance(neighborhoods, Sequence) or len(neighborhoods) != len(rollouts):
            raise ValueError("Per-rollout neighborhoods must match the rollout batch.")
    if use_boc:
        estimates = branch_outcome_credit(rollouts, epsilon)
        outcome_values = tuple(tuple(value.credit for value in row) for row in estimates)
    else:
        grpo = trajectory_credit(rollouts, epsilon)
        outcome_values = tuple(
            (value,) * len(rollout.decisions) for rollout, value in zip(rollouts, grpo, strict=True)
        )
    outcome = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    transition = torch.zeros_like(outcome)
    for i, rollout in enumerate(rollouts):
        positions = mask[i].nonzero(as_tuple=True)[0]
        if positions.numel() != rollout.num_generated_tokens:
            raise ValueError("The response mask and recorded generated-token count disagree.")
        if isinstance(neighborhoods, Mapping):
            neighborhood = neighborhoods.get(rollout.question_id)
        else:
            neighborhood = neighborhoods[i] if neighborhoods is not None else None
        if use_ftc and neighborhood is None:
            raise ValueError("Supply an explicit empty neighborhood for unmapped gold answers.")
        for j, decision in enumerate(rollout.decisions):
            span = positions[decision.token_start : decision.token_end]
            outcome[i, span] = outcome_values[i][j]
            if use_ftc and neighborhood is not None:
                transition[i, span] = neighborhood.credit(decision)
    return TokenCredits(outcome, transition)
