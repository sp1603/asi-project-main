# ASI Project - Traffic Signal Control using Reinforcement Learning

## Overview
This project explores Reinforcement Learning (RL) for traffic signal control using the CityFlow simulation framework.

We implement and evaluate multiple approaches to adaptive traffic signal optimization, including:

- A fixed-time baseline controller
- Tabular Q-learning (implemented from scratch)
- Variations in state representations and reward functions
- Extensions toward Deep Reinforcement Learning (DQN, LSTM)
- LSTM-based vector state representations, where recent traffic sequences are encoded into a learned feature vector and used as input to downstream RL models
- Training and evaluation using Stable Baselines3

The notebook runs a full experimental pipeline:

- Simulation of traffic environments using CityFlow
- Training and evaluation of baseline traffic signal controllers
- Implementation and comparison of multiple Q-learning variants
- Training of Deep Q-Networks (DQN) with LSTM-based state encoding
- Benchmarking against standard RL algorithms (DQN, PPO, A2C)
- Collection of sequential traffic-state data for LSTM training
- Training an LSTM encoder to convert traffic history into compact vector representations
- Using the LSTM-generated vectors as enhanced state inputs for DQN-based learning

The goal of this project is to reduce traffic congestion by learning adaptive traffic signal control policies that outperform traditional fixed-timing approaches.

## Key Metrics

We evaluate performance using:
- Average waiting time
- Total vehicle delay
- Average travel time
- Cumulative reward

## Repository Structure
This project is organized by functionality and experiment type. Each folder has a specific role in the RL pipeline.

The `CityFlow/` folder contains the external traffic simulator used for the project. It stores the road network, traffic flow, and configuration files that define the simulation. The project uses CityFlow to generate vehicles, simulate traffic movement, and apply traffic signal changes.

The `env/` folder contains the custom Gymnasium environment, mainly `cityflow_env.py`. This file wraps CityFlow into a reinforcement learning environment with `reset()` and `step(action)` methods. It defines the observation space, action space, reward function, and how traffic-light phases are applied.

The `scripts/` folder contains the runnable experiment code. This includes baseline controllers, Q-learning experiments, Stable-Baselines experiments, DQN/LSTM extensions, and utility scripts for running simulations. Most project experiments are launched from this folder.

The `scripts/baseline/` folder contains fixed-time traffic signal controllers. These scripts do not use learning; they alternate or control phases using simple rules. Their results are used as a comparison point for the reinforcement learning methods.

The `scripts/q_learning_experiments/` folder contains the main tabular Q-learning experiments. Its subfolders organize different state and reward designs, such as bucketed states, capped states, versions with or without traffic phase information, and different reward functions.

The `scripts/stable_baselines/` folder contains deep reinforcement learning experiments using Stable-Baselines3. These scripts are used for models such as DQN or PPO and help compare tabular Q-learning against neural-network-based RL methods.

The `scripts/dqn_lstm/` folder contains advanced temporal experiments. These are meant to explore whether using previous traffic states through an LSTM can improve the agent’s understanding of congestion trends over time.

The `scripts/lstm_vector/` folder contains LSTM-based experiments for learning temporal representations from traffic history, along with training and evaluation scripts. These are used to evaluate whether richer temporal embeddings improve traffic signal control compared to basic queue-based states.

The `results/` folder stores CSV outputs from experiments. These files include rewards, waiting times, travel times, and comparison summaries. They are used later for analysis and plotting.

The `figures/` folder stores generated graphs and charts. These include reward curves, waiting-over-time plots, and baseline-versus-RL comparisons for the final report and presentation.

The `logs/` folder stores training or debugging logs from experiment runs. This helps track what happened during long runs and makes debugging easier.

The `models/` folder stores trained models when applicable, especially for deep RL methods such as DQN. This allows trained agents to be reused or evaluated later without retraining.

The `requirements.txt` file lists the Python dependencies needed to run the project, such as Gymnasium, NumPy, Pandas, Matplotlib, and Stable-Baselines3.


## Environment Setup -- NEED TO ADD EXPLANATION ON JUPYTER NB

1. Create a Python virtual environment from the root folder. 

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies and upgrade Python packaging tools.

```bash
pip install -r requirements.txt
python -m pip install --upgrade pip setuptools wheel
```

## CityFlow Setup Notes

CityFlow is included as an external simulator dependency and must be installed locally before running the project. Because the original CityFlow build files may not work with newer Python/CMake versions, the setup below should be followed for reproducibility.

### Clone and Install CityFlow
```bash
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
git submodule update --init --recursive
```

### Fix old pybind11/CMake compatibility
If CityFlow fails to build due to old CMake or pybind11 errors, replace the bundled pybind11 version:
```bash
rm -rf extern/pybind11
git clone https://github.com/pybind/pybind11.git extern/pybind11
```

Then update old CMake minimum versions if needed. Search for old CMake requirements:
```bash
grep -R "cmake_minimum_required" .
```

If any build-related files use a version below 3.5, update them to:
cmake_minimum_required(VERSION 3.5)
```

Files that may need this change include:
```bash
CMakeLists.txt
extern/pybind11/CMakeLists.txt
extern/pybind11/tools/pybind11Tools.cmake
```

### Install CityFlow
From inside the CityFlow/ folder:
```bash
python -m pip install .
```

If CMake still raises a policy error, run:
```bash
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" python -m pip install .
```

### Verify Installation 
From the project root: 
```bash
python -c "import cityflow; print('CityFlow installed successfully')"
```

## Usage

### Run a simulation
```bash
python scripts/run_simulation.py
```

### Baseline experiments
```bash
python scripts/baseline/run_baseline.py
python scripts/baseline/plot_baseline_results.py
```

### DQN + LSTM experiments
```bash
python scripts/dqn_lstm/collect_lstm_data.py
python scripts/dqn_lstm/train_lstm.py
python scripts/dqn_lstm/train_dqn_with_lstm.py
python scripts/dqn_lstm/train.py
```

### LSTM Vector experiments
```bash
python -m scripts.lstm_vector.lstm_vector_train
```

### Q-learning experiments
Each folder contains specialized training scripts for different reward and phase handling setups.

Examples:
```bash
python scripts/q_learning_experiments/bucketed_no_phase/train_bucketed_no_phase_default_reward.py
python scripts/q_learning_experiments/bucketed_with_phase/train_bucketed_with_phase_wait_vehicle.py
python scripts/q_learning_experiments/capped_no_phase/train_capped_no_phase_delta_wait.py
python scripts/q_learning_experiments/capped_with_phase/train_capped_with_phase_wait_small_vehicle.py
```

### Stable Baselines 3 training
```bash
python scripts/stable_baselines/train_a2c.py
python scripts/stable_baselines/train_dqn.py
python scripts/stable_baselines/train_ppo.py
```

# asi-project-main
