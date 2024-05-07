#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

It creates Metaworld environmetns. And uses a PPO with 10M
steps.

"""

import click
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer


@click.command()
@click.option('--env-name', type=str, default='push-v2')
@click.option('--seed', type=int, default=np.random.randint(0, 1000))
@click.option('--entropy', type=float, default=0.01)
@wrap_experiment(name_parameters='all', snapshot_mode='gap', snapshot_gap=25)
def ppo_metaworld(ctxt=None, env_name=None, entropy=5e-3, seed=1):
    """Train PPO with Metaworld environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        env_name (str): Name of Meta-World environment to initialize
            this experiment with.
        entropy (float): Coefficient to weigh the entropy reward term by
            when using the max entropy reward.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    stop_entropy_gradient = True
    use_softplus_entropy = False
    set_seed(seed)
    not_in_mw = 'the env_name specified is not a metaworld environment'
    env_name = env_name + '-goal-observable'
    assert env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, not_in_mw
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = env_cls(seed=seed)
    max_path_length = env.max_path_length
    env = normalize(GymEnv(env, max_episode_length=max_path_length),
                    normalize_reward=True)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(128, 128),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            std_share_network=True,
            min_std=0.5,
            max_std=1.5,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(128, 128),
            use_trust_region=True,
        )

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)
        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   discount=0.99,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                   optimizer=FirstOrderOptimizer,
                   optimizer_args=dict(
                       learning_rate=5e-4,
                       max_optimization_epochs=256,
                   ),
                   stop_entropy_gradient=stop_entropy_gradient,
                   entropy_method='max',
                   policy_ent_coeff=entropy,
                   center_adv=False,
                   use_softplus_entropy=use_softplus_entropy,
                   sampler=sampler,
                   use_neg_logli_entropy=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=int(20000000 / (max_path_length * 100)),
                      batch_size=(max_path_length * 100),
                      plot=False)


ppo_metaworld()
