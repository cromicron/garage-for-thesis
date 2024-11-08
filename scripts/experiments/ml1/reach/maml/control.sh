#!/bin/bash
#SBATCH --job-name=nml_reach
#SBATCH --output=logs/maml_reach_parallel_none.txt
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=minor

# Activate the virtual environment
echo "Initializing environment..."
source /home/stud/nalis/thesis/virtualenvs/garage/bin/activate
cd /home/stud/nalis/thesis/garage-for-thesis/constrained_meta_rl_scripts || exit
echo "Running project script..."
python3 maml_parallel.py --env-name reach-v2 --no_train_constraint
echo "Finished Running"