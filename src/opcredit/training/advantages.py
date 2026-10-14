from dataclasses import replace

from opcredit.credit.alignment import build_token_credits


def assign_credits(data, settings):
    records = data.non_tensor_batch["opcredit_record"]
    neighborhoods = data.non_tensor_batch["opcredit_neighborhood"]
    group_ids = data.non_tensor_batch["uid"]
    rollouts = [
        replace(record, question_id=str(uid))
        for record, uid in zip(records, group_ids, strict=True)
    ]
    group_questions = {}
    group_answers = {}
    for uid, record, neighborhood in zip(group_ids, records, neighborhoods, strict=True):
        group = str(uid)
        question = group_questions.setdefault(group, record.question_id)
        if record.question_id != question:
            raise ValueError("A rollout group must contain only one question.")
        if settings.use_ftc:
            gold = group_answers.setdefault(group, neighborhood.gold_entities)
            if neighborhood.gold_entities != gold:
                raise ValueError("Same-question rollouts must share their gold answer entities.")
    credits = build_token_credits(
        rollouts,
        data.batch["response_mask"],
        list(neighborhoods),
        use_boc=settings.use_boc,
        use_ftc=settings.use_ftc,
        epsilon=settings.epsilon,
    )
    data.batch["advantages"] = credits.outcome
    data.batch["returns"] = credits.outcome.clone()
    data.batch["aux_advantage"] = credits.transition
    data.meta_info["opcredit_auxiliary_weight"] = settings.auxiliary_weight
    return data


def install_advantage_hook(settings):
    from verl.trainer.ppo import ray_trainer

    original = ray_trainer.compute_advantage

    def compute_advantage(data, adv_estimator, *args, **kwargs):
        if adv_estimator == "opcredit":
            return assign_credits(data, settings)
        return original(data, adv_estimator, *args, **kwargs)

    ray_trainer.compute_advantage = compute_advantage
