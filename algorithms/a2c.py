"""
a2c_no_branching.py

Trains an A2C model on the RebalancingEnv using the default MlpPolicy,
i.e. no specialized "branching" architecture.
"""

import os
import sys

# Adjust if needed to import your environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.append(PARENT_DIR)

from my_envs.congestion import RebalancingEnv
from src.custom_logging import CustomTensorboardCallback
# Imported here for reference, not used in this particular script
from src.large_a2c import BranchingLargeA2CPolicy

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv


def run_a2c_no_branching():
    """
    Train and evaluate an A2C model on the RebalancingEnv using the default MlpPolicy.
    No specialized 'branching' architecture is utilized.
    """
    # Create environment and wrap in DummyVecEnv for stable-baselines3
    env = RebalancingEnv(
        grid_size=4,
        num_taxis=4,
        timesteps=150,
        demands_file="data/od_demands.json",
        taxi_file="data/taxis.json",
        edge_file="data/edge_file.json"
    )
    vec_env = DummyVecEnv([lambda: env])

    # Build the A2C model with desired hyperparameters
    model = A2C(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=7e-4,
        gamma=0.99,
        ent_coef=0.05,   # Encourage exploration
        vf_coef=0.05,    # Lower value function weight
        n_steps=256,     # Rollout steps
        gae_lambda=0.95,
        device="cpu",
        verbose=1,
        tensorboard_log="./training_logs/deep_branching_a2c/"
    )

    # Train with a custom TensorBoard callback
    callback = CustomTensorboardCallback()
    model.learn(total_timesteps=3_000_000, callback=callback)
    print("Model training complete.")

    # Evaluate the final model
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print("=== A2C Without Branching ===")
    print(f"Total Reward:       {total_reward:.2f}")
    print(f"Trips Completed:    {info['trips_completed']}")
    print(f"Total Travel Time:  {info['total_travel_time']}")
    print(f"Unserved Demand:    {info['unserved_demand']}")


if __name__ == "__main__":
    run_a2c_no_branching()
