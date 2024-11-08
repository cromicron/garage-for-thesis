#!/bin/bash
#SBATCH --job-name=maml_n_rel
#SBATCH --output=logs/const_maml_ml10_relative_scaled_none.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=minor

# Activate the virtual environment
echo "Initializing environment..."
source /home/stud/nalis/thesis/virtualenvs/garage/bin/activate
cd /home/stud/nalis/thesis/garage-for-thesis/constrained_meta_rl_scripts || exit
echo "Running project script..."
python3 constrained_maml_trpo_metaworld_ml10_normalized.py --constraint_mode relative --scale_adv --entropy_method regularized --no_train_constraint --no_const_in_obs --epochs 8000
echo "Finished Running"
