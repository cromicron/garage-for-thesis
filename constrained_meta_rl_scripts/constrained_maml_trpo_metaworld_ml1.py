#!/usr/bin/env python3
"""Train Constrained Meta-Rl on metaworld pick-place env"""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld_constrained as metaworld
import torch

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer
import wandb


# yapf: enable
@click.command()
@click.option('--env-name', type=str, default='pick-place-v2')
@click.option('--seed', type=int, default=1)
@click.option('--epochs', type=int, default=2000)
@click.option('--rollouts_per_task', type=int, default=10)
@click.option('--meta_batch_size', type=int, default=25)
@click.option('--inner_lr', default=1e-4, type=float)
@click.option('--lr_lagrangian', default=3e-2, type=float)
@click.option('--lagrangian', default=30.0, type=float)
@click.option('--constraint_mode', default="relative", type=str)
@click.option('--constraint_size', default=0.03, type=float)
@click.option('--w_and_b', default=True, type=bool)
@wrap_experiment(
    snapshot_mode='last',
    name_parameters='passed',
    archive_launch_repo = False,
    use_existing_dir=True,
)
def constrained_maml_trpo_metaworld_ml1(
    ctxt,
    env_name,
    seed,
    epochs,
    rollouts_per_task,
    meta_batch_size,
    inner_lr,
    lr_lagrangian,
    lagrangian,
    constraint_mode,
    constraint_size,
    w_and_b
):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        env_name (str): Name of Meta-World environment to initialize
            this experiment with.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        rollouts_per_task (int): Number of rollouts per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): learning rate to use for the inner TRPO agent.
            This hyperparameter is typically the one to tune when tuning
            your MAML.
        lagrangian (float): Initial value for lagrangian multiplier
        w_and_b (bool): Whether to log to weights and biases

    """
    set_seed(seed)

    ml1 = metaworld.ML1(env_name, constraint_mode=constraint_mode, constraint_size=constraint_size)
    constructor_args = {"constraint_mode": ml1.constraint_mode, "constraint_size":ml1.constraint_size}
    tasks = MetaWorldTaskSampler(ml1, 'train', constructor_args=constructor_args)
    env_cl = tasks.sample(1)[0]
    env = env_cl()
    test_sampler = SetTaskSampler(
        MetaWorldSetTaskEnv,
        env=MetaWorldSetTaskEnv(ml1, 'test', constructor_args=constructor_args),
        constructor_args={"constructor_args": constructor_args}
    )
    num_test_envs = 5

    policy = GaussianMLPPolicy(env_spec=env.spec,
                               hidden_sizes=(128, 128),
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=torch.tanh,
                               min_std=0.5,
                               max_std=1.5,
                               std_mlp_type='share_mean_std')
    policy.register_parameter(
        "lagrangian",
        torch.nn.Parameter(torch.tensor(lagrangian)))
    # In lagrangian methods, there are two separate advantages
    # thus two value functions necessary
    value_function = LinearFeatureBaseline(env_spec=env.spec)
    value_function_const = LinearFeatureBaseline(
        env_spec=env.spec,
        name="LinearFeatureBaselineConstraints")

    meta_evaluator = MetaEvaluator(test_task_sampler=test_sampler,
                                   n_exploration_eps=rollouts_per_task,
                                   n_test_tasks=num_test_envs * 2,
                                   n_test_episodes=10,
                                   w_and_b=w_and_b)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=meta_batch_size)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        sampler=sampler,
        task_sampler=tasks,
        value_function=value_function,
        value_function_const=value_function_const,
        meta_batch_size=meta_batch_size,
        discount=0.99,
        gae_lambda=1.,
        inner_lr=inner_lr,
        num_grad_updates=1,
        meta_evaluator=meta_evaluator,
        entropy_method='max',
        policy_ent_coeff=5e-5,
        stop_entropy_gradient=True,
        center_adv=False,
        constraint=True,
        constraint_threshold=0.001,
        lr_constraint=lr_lagrangian,
        w_and_b=w_and_b,
    )
    if w_and_b:
        wandb.init(project=f"constrained-maml-ml1-{env_name}",config={
            "inner_rl": inner_lr,
            "meta_batch_size": meta_batch_size,
            "discount": 0.99,
            "gae_lambda": 1,
            "num_grad_updates": 1,
            "policy_ent_coeff": 5e-5,
            "rollouts_per_task": rollouts_per_task,
            "lagrangian_init": lagrangian,
            "lr_lagrangian": lr_lagrangian,
            "constraint_mode": constraint_mode,
            "constraint_size": constraint_size,
            "constraint_threshold": 0.001,
            "environment": "pick-place",

        })
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=rollouts_per_task * env.spec.max_episode_length)


constrained_maml_trpo_metaworld_ml1()
