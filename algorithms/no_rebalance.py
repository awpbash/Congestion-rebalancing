"""
no_rebalance.py

Implements a baseline scenario with no rebalancing strategy:
 - Each free taxi stays put every timestep (action=0).
"""

import sys
import os

# Ensure we can import from parent directory (e.g., for custom environment)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.append(PARENT_DIR)

from my_envs.congestion import CongestionEnv
# from my_envs.rebalancing_env import RebalancingEnv


def run_no_rebalance():
    """
    Runs the environment with no rebalancing:
    - All taxis remain stationary (action=0) each step.
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
        # Each taxi chooses 0 (stay)
        actions = [0] * env.num_taxis
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        if done or truncated:
            break

    # Final metrics
    print("=== No Rebalancing Results ===")
    print(f"Total Reward:         {total_reward:.2f}")
    print(f"Trips Completed:      {info['trips_completed']}")
    print(f"Total Travel Time:    {info['total_travel_time']}")
    print(f"Current Unmet Demand: {info['unserved_demand']}")

    
if __name__ == "__main__":
    run_no_rebalance()
