#!/bin/bash

# Move to the directory where the script is located
cd "$(dirname "$0")" || exit

# Navigate to the project root from the current script location (four levels up)
PROJECT_ROOT=$(cd ../../../../.. && pwd)

# Print the calculated PROJECT_ROOT for verification
echo "Project root directory: $PROJECT_ROOT"

# Check if the virtual environment exists in the project root; if not, prompt the user
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Virtual environment not found in the project root. Please create it in the 'venv' folder."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Navigate to the directory containing maml_ml1.py
cd "$PROJECT_ROOT/constrained_meta_rl_scripts" || exit

# Run the project script
echo "Running project script..."
python3 maml_ml1.py --epochs 4000 --no_train_constraint --env-name reach-v2

echo "Finished Running"
