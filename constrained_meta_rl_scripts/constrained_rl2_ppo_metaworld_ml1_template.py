
#!/usr/bin/env python3
"""Train Constrained Meta-RL on metaworld pick-place env"""

import argparse
import metaworld_constrained as metaworld
import torch
import os
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler, SetTaskSampler)
from garage.experiment.rl2_meta_evaluator import RL2MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.sampler import RaySampler
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.policies import GaussianGRUPolicy
from garage.trainer import Trainer
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["RAY_memory_usage_threshold"] = ".98"
@wrap_experiment(
    snapshot_mode="last",
    archive_launch_repo = False,
    use_existing_dir=True,
    name="rl2_ml1",
)
def main(
    ctxt,
    env_name,
    seed,
    epochs,
    episodes_per_task,
    meta_batch_size,
    inner_lr,
    train_constraint,
    lr_lagrangian,
    lagrangian,
    constraint_mode,
    constraint_size,
    w_and_b,
    entropy_coefficient=5e-6,
    n_epochs_per_eval=5,
    gradient_clip = None,
):

    """Set up environment and algorithm and run the task."""
    set_seed(seed)
    ml1 = metaworld.ML1(
        env_name,
        seed,
        constraint_mode=constraint_mode,
        constraint_size=constraint_size,
    )
    constructor_args = {
        "constraint_mode": ml1.constraint_mode,
        "constraint_size": ml1.constraint_size,
        "include_const_in_obs": ml1.include_const_in_obs,
    }

    # no need to normalize the reward, because it's the same env
    n_constraints = 1 if train_constraint  else 0
    tasks = MetaWorldTaskSampler(
        ml1, 'train',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True),
                              n_constraints=n_constraints), constructor_args=constructor_args)

    test_tasks = MetaWorldTaskSampler(
        ml1, 'test',
        lambda env, _: RL2Env(normalize(env, normalize_reward=True),
                              n_constraints=n_constraints), constructor_args=constructor_args)

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
        load_weights=False,
        weights_dir=f"saved_models/rl2_ml1_constrained_pick_place/rl_2_gru.pth"
    )
    baseline = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        load_weights=False,
        weights_dir=f"saved_models/rl2_ml1_constrained_pick_place/baseline.pth"
        )
    baseline.module.to(device=device, dtype=torch.float64)

    if train_constraint:
        policy.register_parameter(
            "lagrangian",
            torch.nn.Parameter(torch.tensor(lagrangian))
        )
        value_function_const = LinearFeatureBaseline(
            env_spec=env.spec,
            name="LinearFeatureBaselineConstraints")
    else:
        value_function_const = None


    envs = tasks.sample(meta_batch_size)

    sampler = RaySampler(
        agents=policy,
        envs=envs,
        seed=seed +1,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=meta_batch_size,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episodes_per_task))
    test_envs = test_tasks.sample(10)
    test_task_sampler = RaySampler(
        agents=policy,
        envs=test_envs,
        seed=seed+2,
        max_episode_length=env_spec.max_episode_length,
        is_tf_worker=False,
        n_workers=10,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episodes_per_task))
    meta_evaluator = RL2MetaEvaluator(
        sampler=test_task_sampler,
        task_sampler=test_tasks,
        n_exploration_eps=1,
        n_test_tasks=10,
        n_test_episodes=episodes_per_task,
        w_and_b=w_and_b
    )

    optimizer_args_policy = dict(batch_size=32,
                                 max_optimization_epochs=10,
                                 learning_rate=5e-4,
                                 gradient_clip_norm=gradient_clip,
                                 load_state=False,
                                 state_dir="saved_models/rl2_ml1_constrained_pick_place/optimizers")

    optimizer_args_baseline = dict(batch_size=128,
                                 max_optimization_epochs=10,
                                 learning_rate=5e-4,
                                 load_state=False,
                                 state_dir="saved_models/rl2_ml1_constrained_pick_place/optimizers")

    trainer = Trainer(ctxt)
    algo = RL2PPO(meta_batch_size=meta_batch_size,
                  task_sampler=tasks,
                  env_spec=env_spec,
                  policy=policy,
                  baseline=baseline,
                  constraint=True,
                  train_constraint=train_constraint,
                  constraint_threshold=0.001,
                  lr_constraint=lr_lagrangian,
                  baseline_const=value_function_const,
                  sampler=sampler,
                  discount=0.99,
                  gae_lambda=0.95,
                  lr_clip_range=0.2,
                  optimizer_args_policy=optimizer_args_policy,
                  optimizer_args_baseline=optimizer_args_baseline,
                  stop_entropy_gradient=True,
                  entropy_method='max',
                  policy_ent_coeff=entropy_coefficient,
                  center_adv=False,
                  meta_evaluator=meta_evaluator,
                  episodes_per_trial=episodes_per_task,
                  use_neg_logli_entropy=True,
                  n_epochs_per_eval=n_epochs_per_eval,
                  w_and_b=w_and_b,
                  render_every_i=None,
                  save_weights=True,
                  state_dir=f"saved_models/rl2_ml1_constrained_{env_name}_condition_{constraint_mode}_train_constraint={train_constraint}"
                  )

    trainer.setup(algo, envs)

    if w_and_b:
        wandb.init(project=f"constrained-rl2-ml1-{env_name}", config={
            "inner_rl": inner_lr,
            "meta_batch_size": meta_batch_size,
            "discount": 0.99,
            "gae_lambda": 1,
            "policy_ent_coeff": 5e-5,
            "rollouts_per_task": episodes_per_task,
            "lagrangian_init": lagrangian,
            "lr_lagrangian": lr_lagrangian,
            "constraint_mode": constraint_mode,
            "constraint_size": constraint_size,
            "constraint_threshold": 0.001,
            "environment": env_name,
        })

    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task *
                  env_spec.max_episode_length * meta_batch_size,
                  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Constrained RL2 on MetaWorld environments.')
    parser.add_argument('--env-name', type=str, default='pick-place-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--episodes_per_task', type=int, default=10)
    parser.add_argument('--meta_batch_size', type=int, default=25)
    parser.add_argument('--inner_lr', type=float, default=1e-4)

    parser.add_argument('--no_train_constraint', dest= 'train_constraint', action='store_false')
    parser.set_defaults(train_constraint=True)

    parser.add_argument('--lr_lagrangian', type=float, default=5e-1)
    parser.add_argument('--lagrangian', type=float, default=1.0)
    parser.add_argument('--constraint_mode', type=str, default='relative')
    parser.add_argument('--constraint_size', type=float, default=0.03)

    parser.add_argument('--n_epochs_per_eval', type=int, default=10)
    parser.add_argument('--gradient_clip', type=float)

    parser.add_argument('--w_and_b', dest= 'w_and_b', action='store_true')
    parser.add_argument('--no_w_and_b', dest='w_and_b', action='store_false')
    parser.set_defaults(w_and_b=True)
    kwargs = parser.parse_args()
    env_name = kwargs.env_name
    constraint_mode = kwargs.constraint_mode
    train_constraint = kwargs.train_constraint
    experiment_name = f"constrained_rl2_{env_name}_{constraint_mode}_train_constraint={train_constraint}"
    experiment_overrides = {"name": experiment_name}
    main(experiment_overrides, **vars(kwargs))



