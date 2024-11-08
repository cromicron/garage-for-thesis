#!/bin/bash
#SBATCH --job-name=rl2_nrel
#SBATCH --output=logs/const_rl2_ml10_relative_none_no_obs_valid.txt
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
python3 constrained_rl2_ppo_metaworld_ml10.py --constraint_mode relative --no_train_constraint --no_const_in_obs --epochs 20000
echo "Finished Running"
