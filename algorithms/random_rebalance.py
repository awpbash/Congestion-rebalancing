"""
random_rebalance.py

Implements a 'random rebalancing' strategy in RebalancingEnv:
 - Each taxi samples a random action from [0..4] each step:
   0=stay, 1=up, 2=down, 3=left, 4=right.
 - If a taxi is traveling, the environment ignores that action until the taxi is free.
"""

import sys
import os
import numpy as np

# Set a seed for reproducibility (optional)
np.random.seed(0)

# Adjust import path to locate the environment module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.append(PARENT_DIR)

from my_envs.congestion import CongestionEnv
# from my_envs.rebalancing_env import RebalancingEnv


def run_random_rebalance():
    """
    Runs the environment with a random rebalancing strategy:
    - Each taxi picks an action uniformly at random from [0..4].
    """
    env = CongestionEnv(
        grid_size=4,
        num_taxis=20,
        timesteps=150,
        demands_file="data/od_demands.json",
        taxi_file="data/taxis.json",
        edge_file="data/edge_file.json"
    )

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Each taxi chooses a random action in [0..4]
        actions = np.random.randint(low=0, high=5, size=env.num_taxis)
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        if done or truncated:
            break

    # Final metrics
    print("=== Random Rebalancing Results ===")
    print(f"Total Reward:         {total_reward:.2f}")
    print(f"Trips Completed:      {info['trips_completed']}")
    print(f"Total Travel Time:    {info['total_travel_time']}")
    print(f"Current Unmet Demand: {info['unserved_demand']}")


if __name__ == "__main__":
    run_random_rebalance()
