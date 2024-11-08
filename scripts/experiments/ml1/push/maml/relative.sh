#!/bin/bash
#SBATCH --job-name=mml_push
#SBATCH --output=logs/maml_push_parallel.txt
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
python3 maml_parallel.py --env-name push-v2
echo "Finished Running"