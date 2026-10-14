# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Adapted from veRL 0.7.1: joint OpCredit loss and sequence-mean accumulation.

from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

from opcredit.credit.alignment import TokenCredits
from opcredit.credit.objective import ObjectiveConfig, opcredit_loss


class OpCreditActor(DataParallelPPOActor):
    def update_policy(self, data):
        if self.config.loss_agg_mode != "seq-mean-token-mean":
            raise ValueError("OpCredit requires sequence-mean token averaging.")
        if self.use_prefix_grouper or self.ulysses_sequence_parallel_size != 1:
            raise ValueError("This adapter uses text-only FSDP with sequence parallelism 1.")
        if self.config.clip_ratio_low != self.config.clip_ratio_high:
            raise ValueError("Use symmetric policy-ratio clipping.")
        self.actor_module.train()
        settings = ObjectiveConfig(
            clip_ratio=self.config.clip_ratio_low,
            dual_clip=self.config.clip_ratio_c,
            auxiliary_weight=data.meta_info["opcredit_auxiliary_weight"],
            kl_weight=self.config.kl_loss_coef if self.config.use_kl_loss else 0.0,
            entropy_weight=self.config.entropy_coeff,
        )
        keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "aux_advantage",
        ]
        if settings.kl_weight:
            keys.append("ref_log_prob")
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        data = data.select(batch_keys=keys, non_tensor_batch_keys=[])
        metrics = {
            f"actor/{name}": []
            for name in ("outcome_loss", "ftc_loss", "kl_loss", "entropy", "grad_norm")
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in data.split(self.config.ppo_mini_batch_size):
                if self.config.use_dynamic_bsz:
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=self.config.ppo_max_token_len_per_gpu
                    )
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()
                totals = [0.0] * 4
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    inputs = {**micro_batch.batch, "pad_token_id": pad_token_id}
                    outputs = self._forward_micro_batch(
                        inputs,
                        temperature=temperature,
                        calculate_entropy=bool(settings.entropy_weight),
                    )
                    terms = opcredit_loss(
                        outputs["log_probs"],
                        inputs["old_log_probs"],
                        TokenCredits(inputs["advantages"], inputs["aux_advantage"]),
                        inputs["response_mask"],
                        reference_log_probs=inputs.get("ref_log_prob"),
                        entropy=outputs.get("entropys"),
                        config=settings,
                    )
                    scale = len(micro_batch) / len(mini_batch)
                    loss = terms.total * scale
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    for index, value in enumerate(
                        (terms.outcome, terms.transition, terms.kl, terms.entropy)
                    ):
                        totals[index] += value.detach().item() * scale
                grad_norm = self._optimizer_step()
                for name, value in zip(
                    ("outcome_loss", "ftc_loss", "kl_loss", "entropy"), totals, strict=True
                ):
                    metrics[f"actor/{name}"].append(value)
                metrics["actor/grad_norm"].append(grad_norm.detach().item())
        self.actor_optimizer.zero_grad()
        return metrics
