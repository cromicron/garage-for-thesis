#!/usr/bin/env python3
"""Example script to run RL2 in ML10."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld_constrained as metaworld
import math
import torch
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.rl2_meta_evaluator import RL2MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.value_functions import (
    GaussianMLPValueFunction, GammaMLPValueFunction)
from garage.torch.policies import GaussianGRUPolicy
from garage.trainer import Trainer
import wandb
# yapf: enable

device = "cuda" if torch.cuda.is_available() else "cpu"

@click.command()
@click.option('--env-name', type=str, default='pick-place-v2')
@click.option('--seed', default=1)
@click.option('--meta_batch_size', default=25)
@click.option('--n_epochs', default=4000)
@click.option('--episode_per_task', default=10)
@wrap_experiment(
    snapshot_mode='none',
    name_parameters='passed',
    archive_launch_repo=False
)
def rl2_ppo_metaworld_ml1(ctxt,
                          env_name,
                          seed,
                          entropy_coefficient=5e-6,
                          meta_batch_size=25,
                          n_epochs=4000,
                          episode_per_task=10):
    """Train RL2 PPO with ml1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        env_name (str): Name of Meta-World environment to initialize
            this experiment with.
        seed (int): Used to seed the random number generator to produce
            determinism.
        entropy_coefficient (float): Coefficient to weigh the entropy reward
            term by when using the max entropy reward.
        meta_batch_size (int): Number of tasks to sample from during each
            meta training epoch.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    constraints = True
    start_epoch = 0
    n_epochs_per_eval = 5
    set_seed(seed)
    w_and_b = True
    load_state = False
    ml1 = metaworld.ML1(env_name)
    tasks = MetaWorldTaskSampler(
        ml1, 'train',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True), n_constraints=1))
    test_tasks = MetaWorldTaskSampler(
        ml1, 'test',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True), n_constraints=1))

    env_updates = tasks.sample(50)
    env = env_updates[0]()

    env_spec = env.spec
    policy = GaussianGRUPolicy(
        name='policy',
        hidden_dim=256,
        env_spec=env_spec,
        state_include_action=False,
        std_share_network=True,
        init_std=1.,
        min_std=0.5,
        max_std=1.5,
        output_nonlinearity=torch.tanh,
        load_weights=load_state,
        weights_dir=f"saved_models/rl2_ml1_constrained_pick_place/rl_2_gru.pth"
    )

    baseline = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        load_weights=load_state,
        weights_dir=f"saved_models/rl2_ml1_constrained_pick_place/baseline.pth"
        )
    baseline.module.to(device=device, dtype=torch.float64)
    if constraints:
        baseline_const = GammaMLPValueFunction(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            load_weights=False,
            weights_dir="saved_models/rl2_ml1_constrained_pick_place/baseline_c.pth"
            )
        baseline_const.module.to(device=device, dtype=torch.float64)
    else:
        baseline_const = None

    envs = tasks.sample(meta_batch_size)
    sampler = RaySampler(
        agents=policy,
        envs=envs,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=meta_batch_size,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episode_per_task))
    test_envs = test_tasks.sample(5)
    test_task_sampler = RaySampler(
        agents=policy,
        envs=test_envs,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=5,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episode_per_task))
    meta_evaluator = RL2MetaEvaluator(
        sampler=test_task_sampler,
        task_sampler=test_tasks,
        n_exploration_eps=episode_per_task,
        n_test_tasks=5,
        n_test_episodes=3,
        start_eval_itr=math.ceil(start_epoch/n_epochs_per_eval),
        w_and_b=w_and_b
    )
    trainer = Trainer(ctxt, start_at=start_epoch)
    algo = RL2PPO(meta_batch_size=meta_batch_size,
                  task_sampler=tasks,
                  env_spec=env_spec,
                  policy=policy,
                  baseline=baseline,
                  lagrangian_start=20,
                  constraint_threshold=0.001,
                  baseline_const=baseline_const,
                  sampler=sampler,
                  discount=0.99,
                  gae_lambda=0.95,
                  lr_clip_range=0.2,
                  optimizer_args=dict(batch_size=32,
                                      max_optimization_epochs=10,
                                      learning_rate=5e-4,
                                      load_state=load_state,
                                      state_dir="saved_models/rl2_ml1_constrained_pick_place/optimizers"),
                  batch_size_baseline=128,
                  stop_entropy_gradient=True,
                  entropy_method='max',
                  policy_ent_coeff=entropy_coefficient,
                  center_adv=False,
                  meta_evaluator=meta_evaluator,
                  episodes_per_trial=episode_per_task,
                  use_neg_logli_entropy=True,
                  n_epochs_per_eval=n_epochs_per_eval,
                  w_and_b=w_and_b,
                  render_every_i=None,
                  save_weights=True
                  )

    trainer.setup(algo, envs)
    if w_and_b:
        wandb.init(project="rl2-pick-place-constraint",
                   config={
                       # Your configuration parameters here
                       "inner_rl": 5e-4,
                       "meta_batch_size": meta_batch_size,
                       "discount": 0.99,
                       "gae_lambda": 1,
                       "num_grad_updates": 1,
                       "policy_ent_coeff": 5e-5,
                       "lr_lagrangian": 0.01,
                       "lagrangian_start": 20,
                       "constraint_threshold": 0.001,
                       "episode_per_task": episode_per_task
                       # Additional parameters can be added here
                   })
    trainer.train(n_epochs=n_epochs-start_epoch,
                  batch_size=episode_per_task *
                  env_spec.max_episode_length * meta_batch_size,
                  )



rl2_ppo_metaworld_ml1()
