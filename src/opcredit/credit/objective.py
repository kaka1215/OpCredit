from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from opcredit.credit.alignment import TokenCredits, validate_mask


@dataclass(frozen=True)
class ObjectiveConfig:
    clip_ratio: float
    dual_clip: float
    auxiliary_weight: float
    kl_weight: float
    entropy_weight: float

    def __post_init__(self) -> None:
        values = (
            self.clip_ratio,
            self.dual_clip,
            self.auxiliary_weight,
            self.kl_weight,
            self.entropy_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Objective settings must be finite.")
        if not 0 <= self.clip_ratio < 1 or self.dual_clip <= 1:
            raise ValueError("Require 0 <= clip_ratio < 1 and dual_clip > 1.")
        if min(self.auxiliary_weight, self.kl_weight, self.entropy_weight) < 0:
            raise ValueError("Objective weights must be nonnegative.")


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    outcome: torch.Tensor
    transition: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor


def dual_clipped_surrogate(
    ratio: torch.Tensor, advantage: torch.Tensor, config: ObjectiveConfig
) -> torch.Tensor:
    advantage = advantage.detach()
    clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    return torch.where(
        advantage < 0,
        torch.maximum(surrogate, config.dual_clip * advantage),
        surrogate,
    )


def opcredit_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    credits: TokenCredits,
    response_mask: torch.Tensor,
    *,
    reference_log_probs: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    config: ObjectiveConfig,
) -> LossTerms:
    """Average generated tokens within each trajectory, then average trajectories."""
    mask = validate_mask(response_mask)
    if mask.device != log_probs.device:
        raise ValueError("The response mask and log probabilities must share a device.")
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32

    def prepare(value: torch.Tensor, name: str, *, fixed: bool = False) -> torch.Tensor:
        if value.shape != mask.shape or value.device != mask.device:
            raise ValueError(f"{name} must have the same shape and device as the response mask.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if not bool(torch.isfinite(value[mask]).all()):
            raise ValueError(f"{name} contains non-finite values at generated tokens.")
        if fixed:
            value = value.detach()
        return torch.where(mask, value.to(dtype), 0.0)

    current = prepare(log_probs, "log_probs")
    old = prepare(old_log_probs, "old_log_probs", fixed=True)
    outcome_credit = prepare(credits.outcome, "outcome credit", fixed=True)
    transition_credit = prepare(credits.transition, "transition credit", fixed=True)
    ratio = (current - old).exp()
    if not bool(torch.isfinite(ratio[mask]).all()):
        raise FloatingPointError("Policy ratios overflowed; check rollout log probabilities.")

    def mean(value: torch.Tensor) -> torch.Tensor:
        return (torch.where(mask, value, 0.0).sum(dim=1) / mask.sum(dim=1)).mean()

    outcome = -mean(dual_clipped_surrogate(ratio, outcome_credit, config))
    transition = -mean(transition_credit * current)
    kl = current.new_zeros(())
    if config.kl_weight:
        if reference_log_probs is None:
            raise ValueError("KL regularization requires frozen reference log probabilities.")
        reference = prepare(reference_log_probs, "reference_log_probs", fixed=True)
        log_ratio = reference - current
        kl = mean(torch.expm1(log_ratio) - log_ratio)
        if not bool(torch.isfinite(kl)):
            raise FloatingPointError("KL estimation overflowed.")
    entropy_mean = current.new_zeros(())
    if config.entropy_weight:
        if entropy is None:
            raise ValueError("Entropy regularization requires tokenwise policy entropy.")
        entropy_mean = mean(prepare(entropy, "entropy"))
    total = (
        outcome
        + config.auxiliary_weight * transition
        + config.kl_weight * kl
        - config.entropy_weight * entropy_mean
    )
    return LossTerms(total, outcome, transition, kl, entropy_mean)
