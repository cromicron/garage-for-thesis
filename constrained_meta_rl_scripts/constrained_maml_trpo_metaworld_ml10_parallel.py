
#!/usr/bin/env python3
"""Train Constrained Meta-RL on metaworld pick-place env"""

import argparse
import metaworld_constrained as metaworld
import torch

from garage import wrap_experiment
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.maml_meta_evaluator import MetaEvaluator as MamlEvaluator
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer
import wandb

@wrap_experiment(
    snapshot_mode='last',
    archive_launch_repo = False,
    use_existing_dir=True,
    name="constrained_maml_ml10",
)
def main(
    ctxt,
    seed,
    epochs,
    rollouts_per_task,
    meta_batch_size,
    inner_lr,
    entropy_method,
    normalize_adv,
    train_constraint,
    lr_lagrangian,
    lagrangian,
    constraint_mode,
    constraint_size,
    include_const_in_obs,
    w_and_b
):
    """Set up environment and algorithm and run the task."""
    set_seed(seed)
    ml10 = metaworld.ML10(
        seed,
        constraint_mode=constraint_mode,
        constraint_size=constraint_size,
        include_const_in_obs=include_const_in_obs,
    )
    constructor_args = {
        "constraint_mode": ml10.constraint_mode,
        "constraint_size": ml10.constraint_size,
        "include_const_in_obs": ml10.include_const_in_obs,
    }
    tasks = MetaWorldTaskSampler(ml10, "train", constructor_args=constructor_args)
    env = tasks.sample(10)[0]()

    test_tasks = MetaWorldTaskSampler(ml10, "test", constructor_args=constructor_args)
    num_test_envs = 5
    test_env = test_tasks.sample(num_test_envs)[0]()

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=torch.tanh,
        min_std=0.5,
        max_std=1.5,
        std_mlp_type="share_mean_std",
    )
    if train_constraint:
        policy.register_parameter(
            "lagrangian",
            torch.nn.Parameter(torch.tensor(lagrangian))
        )
    value_function = LinearFeatureBaseline(env_spec=env.spec)
    if train_constraint:
        value_function_const = LinearFeatureBaseline(
            env_spec=env.spec,
            name="LinearFeatureBaselineConstraints")
    else:
        value_function_const = None


    test_sampler = RaySampler(
        agents=policy,
        seed=seed+2,
        envs=test_env,
        max_episode_length=env.spec.max_episode_length,
        n_workers=rollouts_per_task
    )

    meta_evaluator = MamlEvaluator(sampler=test_sampler,
                                   task_sampler=test_tasks,
                                   n_exploration_eps=rollouts_per_task,
                                   n_test_tasks=num_test_envs*2,
                                   n_test_episodes=rollouts_per_task,
                                   w_and_b=w_and_b,
                                   pre_post_prefixes=(
                                       "pre_adaptation/",
                                       "post_adaptation/",
                                   ))

    sampler = RaySampler(agents=policy,
                         seed=seed+1,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=rollouts_per_task)

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
        entropy_method=entropy_method,
        policy_ent_coeff=5e-5,
        stop_entropy_gradient=True,
        center_adv=normalize_adv,
        constraint=True,
        train_constraint=train_constraint,
        constraint_threshold=0.001,
        lr_constraint=lr_lagrangian,
        w_and_b=w_and_b,
        save_state=True,
        state_dir=f"saved_models/p_maml_ml10_constrained_{constraint_mode}_const_in_obs={include_const_in_obs}",
        evaluate_every_n_epochs=5,
    )
    if w_and_b:
        wandb.init(project=f"constrained-maml-ml10", config={
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
            "const_in_obs": include_const_in_obs,
        })
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=rollouts_per_task * env.spec.max_episode_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--rollouts_per_task", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=20)
    parser.add_argument("--inner_lr", type=float, default=1e-4)
    parser.add_argument("--entropy_method", type=str, default="max")
    parser.add_argument("--normalize_adv", dest="normalize_adv", action="store_true")

    parser.add_argument("--no_train_constraint", dest= "train_constraint", action="store_false")

    parser.add_argument("--lr_lagrangian", type=float, default=5e-1)
    parser.add_argument("--lagrangian", type=float, default=1.0)
    parser.add_argument("--constraint_mode", type=str, default='relative')
    parser.add_argument("--constraint_size", type=float, default=0.03)
    parser.add_argument("--no_const_in_obs", dest="include_const_in_obs",
                        action="store_false")

    parser.add_argument("--w_and_b", dest= "w_and_b", action="store_true")
    parser.add_argument("--no_w_and_b", dest="w_and_b", action="store_false")
    parser.set_defaults(
        entropy_method="max",
        normalize_adv=False,
        train_constraint=True,
        include_const_in_obs=True,
        w_and_b=True
    )
    kwargs = parser.parse_args()
    constraint_mode = kwargs.constraint_mode
    train_constraint = kwargs.train_constraint
    include_const_in_obs = kwargs.include_const_in_obs
    experiment_name = f"p_constrained_maml_ml10_{constraint_mode}_train_constraint={train_constraint}_include_const_in_obs={include_const_in_obs}"
    experiment_overrides = {"name": experiment_name}
    main(experiment_overrides, **vars(kwargs))



