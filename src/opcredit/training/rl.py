from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from opcredit.agent.kg import FreebaseGraph
from opcredit.agent.modeling import configured_arguments
from opcredit.agent.prompts import SYSTEM_PROMPT, initial_observation
from opcredit.data.prepare import Question, read_records


def prepare_parquet(source, target, endpoint, query_timeout):
    import pyarrow as pa
    import pyarrow.parquet as pq

    graph = FreebaseGraph(endpoint, query_timeout)
    rows = []
    for index, item in enumerate(read_records(source)):
        question = Question.from_dict(item)
        if not question.answer_names or not question.topic_entity:
            raise ValueError("Prepared RL questions require a topic and reference answers.")
        rows.append(
            {
                "data_source": question.dataset,
                "agent_name": "opcredit",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": initial_observation(
                            question.question, graph.name(question.topic_entity)
                        ),
                    },
                ],
                "reward_model": {"style": "rule", "ground_truth": list(question.answer_names)},
                "extra_info": {"index": index, "question_record": item},
            }
        )
    if not rows:
        raise ValueError("No RL questions were found.")
    pq.write_table(pa.Table.from_pylist(rows), target)
    return len(rows)


def build_config(args, train_file, agent_file):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    updates = {
        "data": {
            "train_files": str(train_file),
            "val_files": str(train_file),
            "train_batch_size": args.question_batch_size,
            "max_prompt_length": args.max_prompt_tokens,
            "max_response_length": args.max_response_tokens,
            "seed": args.seed,
            "return_raw_chat": True,
        },
        "actor_rollout_ref": {
            "model": {"path": args.model, "enable_gradient_checkpointing": True},
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": args.question_batch_size,
                "ppo_micro_batch_size_per_gpu": args.micro_batch_size,
                "ppo_epochs": args.ppo_epochs,
                "loss_agg_mode": "seq-mean-token-mean",
                "optim": {"lr": args.learning_rate},
                "clip_ratio": args.clip_ratio,
                "clip_ratio_low": args.clip_ratio,
                "clip_ratio_high": args.clip_ratio,
                "clip_ratio_c": args.dual_clip,
                "use_kl_loss": bool(args.kl_weight),
                "kl_loss_coef": args.kl_weight,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": args.entropy_weight,
                "use_torch_compile": False,
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": args.micro_batch_size},
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "n": args.rollouts,
                "temperature": args.temperature,
                "tensor_model_parallel_size": args.tensor_parallel_size,
                "log_prob_micro_batch_size_per_gpu": args.micro_batch_size,
                "max_model_len": args.max_prompt_tokens + args.max_response_tokens,
                "multi_turn": {"enable": True},
                "agent": {
                    "default_agent_loop": "opcredit",
                    "num_workers": args.agent_workers,
                    "agent_loop_config_path": str(agent_file),
                },
            },
        },
        "algorithm": {"adv_estimator": "opcredit", "use_kl_in_reward": False},
        "trainer": {
            "n_gpus_per_node": args.gpus,
            "nnodes": args.nodes,
            "total_training_steps": args.steps,
            "save_freq": args.save_steps,
            "val_before_train": False,
            "test_freq": -1,
            "logger": ["console"],
            "project_name": "opcredit",
            "experiment_name": "kgqa",
            "default_local_dir": str(args.output),
            "use_legacy_worker_impl": "enable",
            "resume_mode": "resume_path" if args.resume else "disable",
            "resume_from_path": args.resume,
        },
        "opcredit": {
            "endpoint": args.endpoint,
            "embedding_model": args.embedding_model,
            "query_timeout": args.query_timeout,
            "relation_candidates": args.relation_candidates,
            "entity_candidates": args.entity_candidates,
            "neighbor_limit": args.neighbor_limit,
            "mediator_predicate_limit": args.mediator_predicate_limit,
            "ftc_max_predicates": args.ftc_max_predicates,
            "ftc_max_neighbors": args.ftc_max_neighbors,
            "max_hops": args.max_hops,
            "max_turn_tokens": args.max_turn_tokens,
            "max_invalid_operations": args.max_invalid_operations,
            "use_boc": args.use_boc,
            "use_ftc": args.use_ftc,
            "epsilon": args.epsilon,
            "auxiliary_weight": args.auxiliary_weight,
        },
    }
    return OmegaConf.merge(config, updates)


def main():
    parser = argparse.ArgumentParser(description="Train OpCredit with veRL and vLLM.")
    for field in ("model", "data", "endpoint", "embedding-model", "output"):
        parser.add_argument("--" + field, required=True)
    for field in (
        "steps",
        "question-batch-size",
        "rollouts",
        "micro-batch-size",
        "ppo-epochs",
        "max-prompt-tokens",
        "max-response-tokens",
        "max-turn-tokens",
        "max-hops",
        "max-invalid-operations",
        "save-steps",
        "seed",
        "gpus",
        "agent-workers",
    ):
        parser.add_argument("--" + field, type=int, required=True)
    for field in (
        "learning-rate",
        "temperature",
        "clip-ratio",
        "dual-clip",
        "auxiliary-weight",
        "kl-weight",
        "entropy-weight",
    ):
        parser.add_argument("--" + field, type=float, required=True)
    for field in (
        "nodes",
        "tensor-parallel-size",
        "relation-candidates",
        "entity-candidates",
        "neighbor-limit",
        "mediator-predicate-limit",
        "ftc-max-predicates",
        "ftc-max-neighbors",
    ):
        parser.add_argument("--" + field, type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--query-timeout", type=float, required=True)
    parser.add_argument("--use-boc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ftc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    args = configured_arguments(parser)
    from opcredit.credit.objective import ObjectiveConfig

    positive = (
        args.steps,
        args.question_batch_size,
        args.rollouts,
        args.micro_batch_size,
        args.ppo_epochs,
        args.max_prompt_tokens,
        args.max_response_tokens,
        args.max_turn_tokens,
        args.max_hops,
        args.max_invalid_operations,
        args.save_steps,
        args.gpus,
        args.nodes,
        args.tensor_parallel_size,
        args.agent_workers,
        args.relation_candidates,
        args.entity_candidates,
        args.neighbor_limit,
        args.mediator_predicate_limit,
        args.ftc_max_predicates,
        args.ftc_max_neighbors,
        args.learning_rate,
        args.epsilon,
        args.query_timeout,
    )
    if args.temperature <= 0:
        parser.error("Training temperature must be positive.")
    if min(positive) <= 0 or args.seed < 0:
        parser.error("Invalid training, retrieval or parallelism setting.")
    try:
        ObjectiveConfig(
            args.clip_ratio,
            args.dual_clip,
            args.auxiliary_weight,
            args.kl_weight,
            args.entropy_weight,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.gpus % args.tensor_parallel_size:
        parser.error("Tensor parallelism must divide the per-node GPU count.")
    if args.question_batch_size * args.rollouts % (args.gpus * args.nodes * args.micro_batch_size):
        parser.error("The rollout batch must divide evenly across GPU micro-batches.")
    try:
        verl_version = version("verl")
    except PackageNotFoundError as error:
        raise RuntimeError("Install the veRL 0.7.1 source build before starting RL.") from error
    if verl_version != "0.7.1":
        raise RuntimeError("This adapter targets veRL 0.7.1.")
    args.output = Path(args.output).resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or specify --resume.")
    args.output.mkdir(parents=True, exist_ok=True)
    train_file = args.output / "train.parquet"
    count = prepare_parquet(args.data, train_file, args.endpoint, args.query_timeout)
    if count < args.question_batch_size:
        raise ValueError("The dataset is smaller than one question batch.")
    agent_file = args.output / "agent_loop.yaml"
    agent_file.write_text(
        "- name: opcredit\n  _target_: opcredit.agent.rollout.OpCreditAgentLoop\n"
    )
    config = build_config(args, train_file, agent_file)
    import ray
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from opcredit.training.runner import OpCreditTaskRunner

    config = migrate_legacy_reward_impl(config)
    OmegaConf.save(config, args.output / "training_config.yaml")
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(OpCreditTaskRunner))


if __name__ == "__main__":
    main()
