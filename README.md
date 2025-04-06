# Taxi Rebalancing Project

## WORK IN PROGRESS 🏗️🏗️🏗️

This repository explores a **taxi rebalancing** problem in an urban environment, aiming to minimize passenger waiting times and idle cruising by intelligently repositioning taxis in response to demand.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Algorithms & Approaches](#algorithms--approaches)  
4. [Data & Environment](#data--environment)  
5. [Installation & Requirements](#installation--requirements)  
6. [How to Run](#how-to-run)  
7. [Results So Far](#results-so-far)  
8. [Planned Next Steps](#planned-next-steps)  
9. [Project Structure](#project-structure)  
10. [License](#license)

---

## Project Overview

Many urban areas suffer from inefficient taxi distribution: some regions have surplus taxis, while others struggle to meet demand. This project tackles the **rebalancing** challenge by using a reinforcement learning (RL) environment where taxis can move (or "rebalance") to high-demand areas. The primary objective is to reduce both passenger waiting times and total idle mileage.

### Motivation
- **Congestion Reduction**: Fewer empty taxis searching for passengers means less traffic congestion.  
- **Passenger Satisfaction**: Shorter waiting times for ride-hailing services.  
- **Operational Efficiency**: Better resource utilization for fleet operators.

---

## Key Features
- **Reinforcement Learning**: Multiple algorithms (A2C, PPO, DQN, etc.) tested for rebalancing decisions.  
- **Environment Simulation**: Custom environment (`RebalancingEnv`) modeling demand, travel times, and taxi states.  
- **Multiple Demand Profiles**: Ability to use varying demand distributions (Poisson-based, time-of-day patterns, etc.).  
- **Baseline Strategies**: Includes simpler heuristics (No Rebalance, Random, Greedy) for comparative benchmarking.

---

## Algorithms & Approaches

1. **No Rebalancing**  
   - Baseline scenario; all taxis remain where they are, or only move when transporting a passenger.

2. **Random Rebalancing**  
   - Each free taxi chooses a random action among {up, down, left, right, stay}.

3. **Greedy (Top-k) Strategy**  
   - Identify the top-k locations with the highest passenger queues; move taxis step-by-step toward those hotspots.

4. **Reinforcement Learning**  
   - **A2C (No Branching)**: Default MLP policy.  
   - **PPO**: Proximal Policy Optimization with custom or default MLP architecture.  
   - **DQN** (Work in progress): Q-learning approach with function approximation.

---

## Data & Environment

- **Demand Data** (`data/od_demands.json`):  
  Contains origin-destination demand patterns for each timestep. Some variants use Poisson distributions, others are time-of-day dependent.

- **Taxi Data** (`data/taxis.json`):  
  Specifies initial positions and other taxi-related details.

- **Network Graph** (`data/edge_file.json`):  
  Defines node connectivity and travel times between them.

- **Custom Environment** (`my_envs/rebalancing_env.py`):  
  - **Observations**: queue lengths, taxi counts at each node.  
  - **Actions**: discrete moves for each taxi (up, down, left, right, stay).  
  - **Rewards**: Serve passengers quickly and minimize idle travel distance.

---

## Installation & Requirements

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/taxi-rebalancing.git
   cd taxi-rebalancing
   ```# Taxi Rebalancing Project

This repository explores a **taxi rebalancing** problem in an urban environment, aiming to minimize passenger waiting times and idle cruising by intelligently repositioning taxis in response to demand.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Algorithms & Approaches](#algorithms--approaches)  
4. [Data & Environment](#data--environment)  
5. [Installation & Requirements](#installation--requirements)  
6. [How to Run](#how-to-run)  
7. [Results So Far](#results-so-far)  
8. [Planned Next Steps](#planned-next-steps)  
9. [Project Structure](#project-structure)  
10. [License](#license)

---

## Project Overview

Many urban areas suffer from inefficient taxi distribution: some regions have surplus taxis, while others struggle to meet demand. This project tackles the **rebalancing** challenge by using a reinforcement learning (RL) environment where taxis can move (or "rebalance") to high-demand areas. The primary objective is to reduce both passenger waiting times and total idle mileage.

### Motivation
- **Congestion Reduction**: Fewer empty taxis searching for passengers means less traffic congestion.  
- **Passenger Satisfaction**: Shorter waiting times for ride-hailing services.  
- **Operational Efficiency**: Better resource utilization for fleet operators.

---

## Key Features
- **Reinforcement Learning**: Multiple algorithms (A2C, PPO, DQN, etc.) tested for rebalancing decisions.  
- **Environment Simulation**: Custom environment (`RebalancingEnv`) modeling demand, travel times, and taxi states.  
- **Multiple Demand Profiles**: Ability to use varying demand distributions (Poisson-based, time-of-day patterns, etc.).  
- **Baseline Strategies**: Includes simpler heuristics (No Rebalance, Random, Greedy) for comparative benchmarking.

---

## Algorithms & Approaches

1. **No Rebalancing**  
   - Baseline scenario; all taxis remain where they are, or only move when transporting a passenger.

2. **Random Rebalancing**  
   - Each free taxi chooses a random action among {up, down, left, right, stay}.

3. **Greedy (Top-k) Strategy**  
   - Identify the top-k locations with the highest passenger queues; move taxis step-by-step toward those hotspots.

4. **Reinforcement Learning**  
   - **A2C (No Branching)**: Default MLP policy.  
   - **PPO**: Proximal Policy Optimization with custom or default MLP architecture.  
   - **DQN** (Work in progress): Q-learning approach with function approximation.

---

## Data & Environment

- **Demand Data** (`data/od_demands.json`):  
  Contains origin-destination demand patterns for each timestep. Some variants use Poisson distributions, others are time-of-day dependent.

- **Taxi Data** (`data/taxis.json`):  
  Specifies initial positions and other taxi-related details.

- **Network Graph** (`data/edge_file.json`):  
  Defines node connectivity and travel times between them.

- **Custom Environment** (`my_envs/rebalancing_env.py`):  
  - **Observations**: queue lengths, taxi counts at each node.  
  - **Actions**: discrete moves for each taxi (up, down, left, right, stay).  
  - **Rewards**: Serve passengers quickly and minimize idle travel distance.

---

## Installation & Requirements

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/taxi-rebalancing.git
   cd taxi-rebalancing
