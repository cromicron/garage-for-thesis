import argparse
import json
from garage.experiment.experiment import wrap_experiment
from garage.trainer import Trainer
import wandb
import os

project_name = "maml_ml10"
@wrap_experiment(
    snapshot_mode='last',
    archive_launch_repo = False,
    use_existing_dir=False,
    name="maml_ml1",
)
def continue_maml(
    ctxt,
    snapshot_path,
    w_and_b_project,
    w_and_b_run_id,
    snapshot_index,
    n_epochs = None,
):
    """
    Continue training a model using MAML from a saved snapshot, with optional Weights & Biases integration.

    Args:
        ctxt (SnapshotConfig): Configuration context for the experiment.
        snapshot_path (str): Path to the snapshot file.
        w_and_b_project (str): The project name on Weights & Biases.
        w_and_b_run_id (str): The run ID on Weights & Biases, for resuming tracking.
        snapshot_index (str): Optional index to append to the snapshot path, if multiple snapshots.
    """
    resume_w_and_b_run = "must" if w_and_b_run_id is not None else False
    wandb.init(project=w_and_b_project, id=w_and_b_run_id, resume=resume_w_and_b_run)
    trainer = Trainer(ctxt)
    snapshot_path_suffix = "" if snapshot_index is None else "_" + snapshot_index
    trainer.restore(snapshot_path + snapshot_path_suffix)
    trainer.resume(n_epochs=n_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue MAML from snapshot")
    parser.add_argument("--snapshot_path", type=str, required=True)
    parser.add_argument("--w_and_b_project", type=str, required=True)
    parser.add_argument("--w_and_b_run_id", type=str)
    parser.add_argument("--snapshot_index", type=str)
    parser.add_argument("--n_epochs", type=int)
    kwargs = parser.parse_args()

    # Overrides for the experiment - important for snapshotter
    with open(f"{kwargs.snapshot_path}/experiment.json", 'r') as file:
        snapshotter = json.load(file).get("_snapshotter")
    project_name = os.path.basename(kwargs.snapshot_path)
    experiment_override = {
        "name": project_name,
        "snapshot_mode": snapshotter["_snapshot_mode"],
        "snapshot_gap": snapshotter["_snapshot_gap"]
    }
    continue_maml(experiment_override, **vars(kwargs))
