#!/bin/bash
#SBATCH --job-name=rl2_pn
#SBATCH --output=logs/const_rl2_ml1_pickplace_none.txt
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
python3 constrained_rl2_ppo_metaworld_ml1_template.py --no_train_constraint
echo "Finished Running"
