# Reinforcement Learning on Pendulum-v1 Environment

This repository contains an implementation of various **Reinforcement Learning (RL)** algorithms applied to the **Pendulum-v1** environment from OpenAI Gym. The goal of the project is to solve the continuous control task by balancing the pendulum upright using RL techniques.

## Features
- **Environment**: OpenAI Gym's `Pendulum-v1` environment, which simulates a pendulum's dynamics, requiring the agent to learn a control policy to keep it upright.
- **RL Algorithms**:
  - **Deep Q-Learning (DQL)**: Applied to continuous action spaces.
  - **Policy Gradient Methods**: Implementations of policy-based algorithms.
  - **Actor-Critic**: Actor-Critic architecture with separate actor and critic networks.
  - **Proximal Policy Optimization (PPO)**(TODO): Implemented for more stable training in continuous action spaces.

  
## Requirements
- `PyTorch` for building and training neural networks.
- `OpenAI Gym` for environment setup and simulation.
- `NumPy`, `matplotlib` for numerical operations and plotting results.

## Usage
- Train the agent using different RL algorithms and monitor performance via reward plots.
- Explore various hyperparameters like learning rates, decay rates, and network architectures.

## Goal
- The main goal is to minimize the angle and angular velocity of the pendulum, ensuring it remains upright by controlling the torque applied to it.