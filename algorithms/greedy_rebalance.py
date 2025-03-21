"""
greedy_rebalance.py

Implements a 'greedy rebalancing' strategy:
 - Each step, identify nodes with the largest passenger queues.
 - Select the top_k nodes by queue length.
 - For each free taxi, move one step toward the nearest node among those top_k.
 - If the taxi is already on one of the top_k nodes or there's no beneficial move, it stays.
"""

import sys
import os
import numpy as np
import networkx as nx

# Make sure we can import from parent directory (e.g., for custom environment)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.append(PARENT_DIR)

from my_envs.congestion import CongestionEnv
# from my_envs.rebalancing_env import RebalancingEnv


def node_to_xy(node_idx, grid_size):
    """
    Convert a node index to (x, y) coordinates in a grid.
    :param node_idx: The node's index.
    :param grid_size: Width/height of the grid.
    :return: (x, y) coordinates.
    """
    return divmod(node_idx, grid_size)


def xy_to_node(x, y, grid_size):
    """
    Convert (x, y) coordinates to a node index in a grid.
    :param x: Row index.
    :param y: Column index.
    :param grid_size: Width/height of the grid.
    :return: Node index.
    """
    return x * grid_size + y


def one_step_direction(src_node, dst_node, grid_size):
    """
    Determine which discrete action [0..4] moves one step from src_node toward dst_node.
    0 = stay, 1 = up, 2 = down, 3 = left, 4 = right.
    If src_node == dst_node, return 0 (stay).
    """
    if src_node == dst_node:
        return 0  # already there

    sx, sy = node_to_xy(src_node, grid_size)
    dx, dy = node_to_xy(dst_node, grid_size)

    # Try to reduce the Manhattan distance by 1 step
    if sx < dx:
        return 2  # down
    elif sx > dx:
        return 1  # up
    elif sy < dy:
        return 4  # right
    elif sy > dy:
        return 3  # left

    return 0  # fallback


def parse_observation(obs_array, num_nodes):
    """
    Parse the environment observation array into a list of (queue_length, taxis_available).
    :param obs_array: Flattened observation array of length 2 * num_nodes.
                     Format: [q_len(node0), taxis(node0), q_len(node1), taxis(node1), ...]
    :param num_nodes: Number of nodes in the environment.
    :return: A list where each element i is (queue_len_i, taxis_avail_i).
    """
    node_info = []
    for i in range(num_nodes):
        q_len = obs_array[2 * i]
        t_avail = obs_array[2 * i + 1]
        node_info.append((q_len, t_avail))
    return node_info


def run_greedy_topk(top_k=3):
    """
    Run the environment using a Greedy Rebalancing strategy with top_k prioritization.
    :param top_k: Number of nodes with highest queue length to prioritize each step.
    """
    # 1) Create and reset environment
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
        # Parse current observation
        node_data = parse_observation(obs, env.num_nodes)
        # Sort nodes by queue length (descending)
        sorted_nodes = sorted(
            range(env.num_nodes),
            key=lambda i: node_data[i][0],  # index => queue length
            reverse=True
        )
        # Select the top_k nodes with the highest queues
        topk_nodes = sorted_nodes[:top_k]

        # Build an action array for each taxi
        actions = np.zeros(env.num_taxis, dtype=int)  # default "stay"

        # For each taxi, decide on a single-step move toward the nearest top_k node if beneficial
        for i, taxi in enumerate(env.taxis):
            if taxi["remaining_time"] > 0:
                # Taxi is busy, can't move now
                actions[i] = 0
                continue

            src_node = taxi["current_node"]
            # If there's already a queue here, stay to serve it
            if node_data[src_node][0] > 0:
                actions[i] = 0
                continue

            # Otherwise, pick the top_k node that has > 0 queue and is closest by shortest path
            best_node = None
            best_dist = float('inf')
            for nd in topk_nodes:
                if node_data[nd][0] <= 0:
                    continue  # no queue left to serve here
                dist = nx.shortest_path_length(env.graph, source=src_node, target=nd, weight='weight')
                if dist < best_dist:
                    best_dist = dist
                    best_node = nd

            if best_node is not None and best_node != src_node:
                actions[i] = one_step_direction(src_node, best_node, env.grid_size)
            else:
                actions[i] = 0  # no beneficial move

        # Step the environment
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        if done or truncated:
            break

    # Final metrics
    print("=== Greedy Rebalancing Results ===")
    print(f"Top_k = {top_k}")
    print(f"Total Reward:         {total_reward:.2f}")
    print(f"Trips Completed:      {info['trips_completed']}")
    print(f"Total Travel Time:    {info['total_travel_time']}")
    print(f"Current Unmet Demand: {info['unserved_demand']}")


if __name__ == "__main__":
    # Example usage
    run_greedy_topk(top_k=1)
