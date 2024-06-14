#!/usr/bin/env python3
"""Example script to run RL2 in ML10."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import math
import torch
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.rl2_meta_evaluator import RL2MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler as RaySampler
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.policies import GaussianHyperGRUPolicy
from garage.trainer import Trainer
import wandb
# yapf: enable
device = "cuda" if torch.cuda.is_available() else "cpu"

@click.command()
@click.option('--seed', default=1)
@click.option('--entropy_coefficient', type=float, default=5e-6)
@wrap_experiment(
    snapshot_mode='none',
    name_parameters='passed',
    archive_launch_repo = False,
)
def rl2_ppo_metaworld_ml10(ctxt,
                           seed,
                           entropy_coefficient=5e-6,
                           meta_batch_size=10,
                           n_epochs=10000,
                           episode_per_task=10,
                           ):
    """Train RL2 PPO with ML10 environment.

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
    n_epochs_per_eval = 2
    run_in_episodes = 0
    set_seed(seed)
    w_and_b = True
    load_state = False
    ml10 = metaworld.ML10()
    tasks = MetaWorldTaskSampler(
        ml10, 'train',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True)))
    test_tasks = MetaWorldTaskSampler(
        ml10, 'test',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True)))

    env_updates = tasks.sample(10)
    env = env_updates[0]()

    env_spec = env.spec
    policy = GaussianHyperGRUPolicy(
        name='policy',
        hidden_dim=256,
        env_spec=env_spec,
        policy_input_dim=40,
        state_include_action=False,
        min_std=0.5,
        max_std=1.5,
        output_nonlinearity=torch.tanh,
        load_weights=load_state,
    )
    baseline = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        load_weights=load_state
        )
    baseline.module.to(device=device, dtype=torch.float64)

    envs = tasks.sample(meta_batch_size)
    test_envs = test_tasks.sample(10)
    sampler = RaySampler(
        agents=policy,
        envs=envs,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=meta_batch_size,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episode_per_task))
    test_task_sampler = RaySampler(
        agents=policy,
        envs=test_envs,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=10,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episode_per_task))
    meta_evaluator = RL2MetaEvaluator(
        sampler=test_task_sampler,
        task_sampler=test_tasks,
        n_exploration_eps=episode_per_task,
        n_test_tasks=5 * 2,
        n_test_episodes=10,
        start_eval_itr=math.ceil(start_epoch/n_epochs_per_eval),
        w_and_b=w_and_b
    )

    trainer = Trainer(ctxt, start_at=start_epoch)
    algo = RL2PPO(meta_batch_size=meta_batch_size,
                  task_sampler=tasks,
                  env_spec=env_spec,
                  policy=policy,
                  baseline=baseline,
                  sampler=sampler,
                  discount=0.99,
                  gae_lambda=0.95,
                  lr_clip_range=0.2,
                  optimizer_args=dict(batch_size=256,
                                      max_optimization_epochs=10,
                                      learning_rate=5e-4,
                                      load_state=load_state),
                  stop_entropy_gradient=True,
                  entropy_method='max',
                  policy_ent_coeff=entropy_coefficient,
                  center_adv=False,
                  meta_evaluator=meta_evaluator,
                  episodes_per_trial=episode_per_task,
                  use_neg_logli_entropy=True,
                  n_epochs_per_eval=n_epochs_per_eval,
                  save_weights=False,
                  w_and_b=w_and_b,
                  run_in_episodes=run_in_episodes,
                  render_every_i=None
                  )

    trainer.setup(algo, envs)
    if w_and_b:
        wandb.init(project="rl2_hyper_ml10",
                   config={
                       # Your configuration parameters here
                       "inner_rl": 5e-4,
                       "meta_batch_size": meta_batch_size,
                       "latent": 10,
                       "hidden": 256,
                       "discount": 0.99,
                       "gae_lambda": 1,
                       "num_grad_updates": 1,
                       "policy_ent_coeff": 5e-5,
                       "episode_per_task": episode_per_task
                       # Additional parameters can be added here
                   })
    trainer.train(n_epochs=n_epochs-start_epoch,
                  batch_size=episode_per_task *
                  env_spec.max_episode_length * meta_batch_size,
                  )



rl2_ppo_metaworld_ml10()
