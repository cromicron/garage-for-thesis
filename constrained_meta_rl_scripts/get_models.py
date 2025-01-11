import argparse
import json
from garage.experiment.experiment import wrap_experiment
from garage.trainer import Trainer
import os
import torch

project_name = "maml_ml1"
@wrap_experiment(
    snapshot_mode='last',
    archive_launch_repo = False,
    use_existing_dir=False,
    name="maml_ml1",
)
def get_maml_model(
    ctxt,
    snapshot_path,
    snapshot_index,
    save_path,
):
    """
    Continue training a model using MAML from a saved snapshot, with optional Weights & Biases integration.

    Args:
        ctxt (SnapshotConfig): Configuration context for the experiment.
        snapshot_path (str): Path to the snapshot file.
        snapshot_index (str): Optional index to append to the snapshot path, if multiple snapshots.
    """
    trainer = Trainer(ctxt)
    snapshot_path_suffix = "" if snapshot_index is None else "_" + snapshot_index
    trainer.restore(snapshot_path + snapshot_path_suffix)
    params = trainer._algo.policy.state_dict()
    torch.save(params, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get MAML model from snapshot")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--snapshot_index", type=str)
    parser.add_argument("--save_path", type=str, required=True)
    kwargs = parser.parse_args()

    # Overrides for the experiment - important for snapshotter
    path_experiments = "/home/stud/nalis/thesis/garage-for-thesis/constrained_meta_rl_scripts/data/local/experiment"
    snapshot_path = os.path.join(path_experiments,kwargs.experiment_name)
    with open(f"{snapshot_path}/experiment.json", 'r') as file:
        snapshotter = json.load(file).get("_snapshotter")
    experiment_override = {
        "name": kwargs.experiment_name,
        "snapshot_mode": snapshotter["_snapshot_mode"],
        "snapshot_gap": snapshotter["_snapshot_gap"]
    }
    get_maml_model(
        experiment_override,
        snapshot_path=snapshot_path,
        snapshot_index=kwargs.snapshot_index,
        save_path=kwargs.save_path,
    )
