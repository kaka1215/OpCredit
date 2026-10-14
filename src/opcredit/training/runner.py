import ray
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from opcredit.training.advantages import install_advantage_hook


class OpCreditWorker(AsyncActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        import verl.workers.actor
        from opcredit.training.actor import OpCreditActor

        verl.workers.actor.DataParallelPPOActor = OpCreditActor
        super().__init__(*args, **kwargs)


class OpCreditTaskRunner(TaskRunner):
    def run(self, config):
        install_advantage_hook(config.opcredit)
        return super().run(config)

    def add_actor_rollout_worker(self, config):
        if (
            config.actor_rollout_ref.actor.strategy != "fsdp"
            or config.trainer.use_legacy_worker_impl != "enable"
        ):
            raise ValueError("Use the veRL FSDP worker configuration provided with OpCredit.")
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(OpCreditWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return OpCreditWorker, RayWorkerGroup
