#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import torch
import wandb
from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler #LocalSampler as RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.trainer import Trainer
import pandas as pd


# yapf: enable
@click.command()
@click.option('--seed', type=int, default=1)
@click.option('--epochs', type=int, default=2000)
@click.option('--rollouts_per_task', type=int, default=10)
@click.option('--meta_batch_size', type=int, default=20)
@click.option('--inner_lr', default=1e-4, type=float)
@wrap_experiment(snapshot_mode='last', name_parameters='passed')
def maml_trpo_metaworld_ml10(ctxt, seed, epochs, rollouts_per_task,
                             meta_batch_size, inner_lr):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        rollouts_per_task (int): Number of rollouts per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): learning rate to use for the inner TRPO agent.
            This hyperparameter is typically the one to tune when tuning
            your MAML.

    """
    """
    wandb.init(project="maml_garage_metaworld_10", config={
        # Your configuration parameters
        "inner_lr": inner_lr,
        "meta_batch_size": meta_batch_size,
        "discount": 0.99,
        "gae_lambda": 1,
        "num_grad_updates": 1,
        "policy_ent_coeff": 5e-5,
        "rollouts_per_task": rollouts_per_task,
    },resume=True)
    df_progress = pd.read_csv("data/local/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=2000_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001/progress.csv")
    for i, row in df_progress.iterrows():
        for col  in row.index:
            if col.endswith("AverageReturn") or col.endswith("/SuccessRate"):
                wandb.log({
                    col: row[col],
                }, step=i)
    """
    trainer = Trainer(ctxt)
    trainer.restore("data/local/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=2000_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001")
    trainer.resume()


maml_trpo_metaworld_ml10()
