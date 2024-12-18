#!/usr/bin/env python3
"""Example script to run RL2 in ml45."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import tensorflow as tf
import wandb
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

# yapf: enable


@click.command()
@click.option('--seed', default=1)
@click.option('--meta_batch_size', default=45)
@click.option('--n_epochs', default=2000)
@click.option('--episode_per_task', default=10)
@wrap_experiment(snapshot_mode='none', name_parameters='passed')
def rl2_ppo_metaworld_ml45(ctxt,
                           seed,
                           entropy_coefficient=5e-6,
                           meta_batch_size=25,
                           n_epochs=4000,
                           episode_per_task=10):
    """Train RL2 PPO with ml45 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        entropy_coefficient (float): Coefficient to weigh the entropy reward
            term by when using the max entropy reward.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    start_epoch = 0
    n_epochs_per_eval = 50
    set_seed(seed)
    w_and_b = True
    with TFTrainer(snapshot_config=ctxt) as trainer:
        ml45 = metaworld.ML45()
        tasks = MetaWorldTaskSampler(
            ml45, 'train',
            lambda env, _: RL2Env(normalize(env, normalize_reward=True)))
        test_task_sampler = SetTaskSampler(
            MetaWorldSetTaskEnv,
            env=MetaWorldSetTaskEnv(ml45, 'test'),
            wrapper=lambda env, _: RL2Env(normalize(env, normalize_reward=True)
                                          ))
        num_test_envs = 5
        meta_evaluator = MetaEvaluator(test_task_sampler=test_task_sampler,
                                       n_exploration_eps=episode_per_task,
                                       n_test_tasks=num_test_envs * 2,
                                       n_test_episodes=10)

        env_updates = tasks.sample(45)
        env = env_updates[0]()

        env_spec = env.spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=256,
                                   env_spec=env_spec,
                                   state_include_action=False,
                                   std_share_network=True,
                                   init_std=1.,
                                   min_std=0.5,
                                   max_std=1.5,
                                   output_nonlinearity=tf.nn.tanh)

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(256, 256),
            use_trust_region=False,
            optimizer=FirstOrderOptimizer,
            optimizer_args=dict(
                batch_size=32,
                max_optimization_epochs=10,
                learning_rate=3e-4,
            ),
        )

        envs = tasks.sample(meta_batch_size)
        sampler = RaySampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2PPO(
            meta_batch_size=meta_batch_size,
            task_sampler=tasks,
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            sampler=sampler,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(batch_size=32,
                                max_optimization_epochs=10,
                                learning_rate=5e-4),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=entropy_coefficient,
            center_adv=False,
            meta_evaluator=meta_evaluator,
            episodes_per_trial=episode_per_task,
            use_neg_logli_entropy=True,
        )

        trainer.setup(algo, envs)
        if w_and_b:
            wandb.init(project="rl2-garage-metaworld_45-tf",
                       config={
                           # Your configuration parameters here
                           "inner_rl": 5e-4,
                           "meta_batch_size": meta_batch_size,
                           "discount": 0.99,
                           "gae_lambda": 1,
                           "num_grad_updates": 1,
                           "policy_ent_coeff": 5e-5,
                           "episode_per_task": episode_per_task
                           # Additional parameters can be added here
                       })
        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task *
                      env_spec.max_episode_length * meta_batch_size)


rl2_ppo_metaworld_ml45()
