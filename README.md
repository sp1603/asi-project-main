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

| Path | Description |
|---|---|
| `CityFlow/` | External traffic simulator and Hangzhou dataset files, including road networks, traffic flows, and simulation configs. |
| `env/` | Custom Gymnasium environment. `cityflow_env.py` wraps CityFlow with `reset()` and `step(action)` methods. |
| `scripts/` | Main runnable experiment scripts for baselines, Q-learning, Stable-Baselines3, DQN/LSTM, and utilities. |
| `scripts/baseline/` | Fixed-time traffic signal controllers used as non-learning baselines. |
| `scripts/q_learning_experiments/` | Tabular Q-learning experiments organized by state representation and reward function. |
| `scripts/stable_baselines/` | Deep RL experiments using Stable-Baselines3 models such as DQN, PPO, and A2C. |
| `scripts/dqn_lstm/` | Temporal DQN/LSTM experiments using previous traffic states to improve learning. |
| `scripts/lstm_vector/` | LSTM representation-learning experiments that convert traffic history into vector state inputs. |
| `results/` | Generated experiment outputs, including training logs, traces, Q-tables, and comparison CSVs. |
| `figures/` | Generated plots and charts for the report and presentation. |
| `logs/` | Training and debugging logs from experiment runs. |
| `models/` | Saved trained models for reuse or later evaluation. |
| `requirements.txt` | Python dependency list, if used separately from `environment.yml`. |
| `environment.yml` | Conda environment file used to reproduce the project setup. |

## Environment Setup

To reproduce the experiments, set up a Conda environment and install CityFlow from source.

### 1. Create and activate Conda environment

```bash
conda env create -f environment.yml
conda activate traffic-rl
```

### 2. Install CityFlow

```bash
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
git submodule update --init --recursive
```

```bash
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
git submodule update --init --recursive

sed -i.bak 's/cmake_minimum_required(VERSION 3.0)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
sed -i.bak 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' extern/pybind11/CMakeLists.txt
sed -i.bak 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' extern/pybind11/tools/pybind11Tools.cmake

rm -rf build dist *.egg-info
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install .
```

```bash
# Verify Installation 
python -c "import cityflow; print('CityFlow installed successfully')"
```

### 3. Add the Conda Environment to Jupyter

```bash
cd .. 
python -m ipykernel install --user --name traffic-rl --display-name "traffic-rl"
jupyter notebook
```

### 4. Go to Traffic_Signal_RL_Reproducibility.ipynb
Run the cells in order in the notebook. The notebook installs/checks dependencies, runs the experiment scripts, and loads result files for comparison.
