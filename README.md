
# Constrained Meta Reinforcement Learning with MetaWorld

This repository is an adaptation of [garage](https://github.com/rlworkgroup/garage), tailored to test MAML and RL² algorithms in constrained meta reinforcement learning settings using [MetaWorld](https://github.com/Farama-Foundation/Metaworld).

## Overview

Developed as part of a master's thesis, this project focuses on implementing and evaluating constrained meta reinforcement learning algorithms. It extends the capabilities of the original garage framework to support training and testing within constrained environments provided by a customized version of MetaWorld.

## Features

- **Constrained Meta-RL Algorithms**: Implementation of MAML and RL² algorithms adapted for constrained environments.
- **Customized MetaWorld Environments**: Utilizes an adjusted version of MetaWorld with specific constraints applied to tasks and environments.
- **Based on Garage PR #2287**: Built upon the branch from [this pull request](https://github.com/rlworkgroup/garage/pull/2287), ensuring compatibility and stability with recent updates.

## Installation

### Prerequisites

- Python 3.8.19 (tested and recommended)
- Git
- **Linux OS** (recommended; Windows users may need to use Windows Subsystem for Linux (WSL) for compatibility)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/cromicron/garage-for-thesis.git
   ```

2. **Set Up a Virtual Environment in the Project Root**

   Navigate to the project root (the folder created by the clone):

   ```bash
   cd garage-for-thesis
   ```

   Create and activate a virtual environment in the root directory:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows using WSL, use the same command
   ```

3. **Install Dependencies**

   With the virtual environment activated, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   This command will automatically install both `metaworld_constrained` and `akro` from their GitHub repositories, along with other dependencies.

## Running Experiments

The experiments are organized under `ml1` and `ml10` directories, each containing tasks and algorithms (MAML and RL²). Scripts for each experiment scenario are located within the `scripts` folder of the project.

## Directory Structure Overview

- `scripts/`
  - `experiments/`
    - `ml1/`
      - `pick_place/`
      - `push/`
      - `reach/`
    - `ml10/`

Each task directory within `ml1` and `ml10` contains subdirectories for `maml` and `rl2`, where the experimental scripts are located.

## ML1 Experiments

Navigate to the appropriate task and algorithm within the `scripts/experiments/ml1/` directory to run the corresponding scripts.

### MAML Algorithm

For tasks like `pick_place`, `push`, or `reach` under MAML:

```bash
cd scripts/experiments/ml1/<task>/maml
./control.sh       # For control scenario
./relative.sh      # For relative scenario
./random.sh        # For random scenario
```

### RL² Algorithm

For the same tasks under RL²:

```bash
cd scripts/experiments/ml1/<task>/rl2
./control.sh       # For control scenario
./relative.sh      # For relative scenario
./random.sh        # For random scenario
```

## ML10 Experiments

Navigate to the `ml10` directory for MAML and RL² experiments, where scripts are tailored for different control scenarios.

### MAML and RL² Algorithms

```bash
cd scripts/experiments/ml10/<algorithm>
./control_relative.sh   # For control relative scenario
./control_random.sh     # For control random scenario
./relative.sh           # For relative scenario
./random.sh             # For random scenario
```

Replace `<task>` with `pick_place`, `push`, or `reach` and `<algorithm>` with `maml` or `rl2` as per your experiment setup.

## Running Scripts

Ensure you have the necessary permissions to execute the scripts:

```bash
chmod +x *.sh   # Grant execution permissions if needed
```

### Note on `control.sh` Script

Each `control.sh` script is designed to:
- Activate the virtual environment located in the root directory (`venv`).
- Run the specified experiment in the `maml_ml1.py` file located in `constrained_meta_rl_scripts`.

Before running `control.sh`, ensure the virtual environment (`venv`) is created in the project root and dependencies are installed.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements.

## References

- [Garage Repository](https://github.com/rlworkgroup/garage)
- [Original MetaWorld](https://github.com/Farama-Foundation/Metaworld)
- [Adjusted MetaWorld Constrained](https://github.com/cromicron/metaworld_constrained)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **rlworkgroup** for the original [garage](https://github.com/rlworkgroup/garage) repository.
- **Farama Foundation** for the original [MetaWorld](https://github.com/Farama-Foundation/Metaworld).
- Supervisors and peers who supported the master's thesis project.
