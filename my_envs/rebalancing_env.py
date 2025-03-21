import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from collections import deque
import json
import torch
import random


class CongestionEnv(gym.Env):
    """
    Multi-taxi rebalancing environment with proactive demand forecasting:
      - Observations include current passenger queues, taxis available, and future demand forecasts.
      - Each taxi chooses from 5 actions: [stay, up, down, left, right].
      - Taxis serve waiting passengers or rebalance themselves proactively.
    """

    def __init__(self, grid_size, num_taxis, timesteps, demands_file="../data/od_demands.json",
                 taxi_file="../data/taxis.json", edge_file="../data/edge_file.json", forecast_horizon=5):
        super().__init__()

        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.num_taxis = num_taxis
        self.timesteps = timesteps
        self.forecast_horizon = forecast_horizon

        # Load demand data
        with open(demands_file, "r") as f:
            self.demands_data = json.load(f)
        self.demands_data = {int(k): v for k, v in self.demands_data.items()}

        # Load taxi positions and edge weights
        with open(taxi_file, "r") as f:
            self.taxi_positions = json.load(f)["taxi_positions"]

        with open(edge_file, "r") as f:
            self.edge_weights = json.load(f)

        # Define observation space: current state + forecasted future demand
        obs_dim = (2 * self.num_nodes) + (self.forecast_horizon * self.num_nodes)
        self.observation_space = spaces.Box(low=0, high=9999, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5] * self.num_taxis)

        # Initialize environment and load all demands
        self._create_environment()
        self.all_demands = self._load_all_demands()

    def _load_all_demands(self):
        """Pre-load all future demands into a list of dictionaries."""
        all_demands = []
        for t in range(self.timesteps + self.forecast_horizon):
            demands_at_t = self.demands_data.get(t, {})
            all_demands.append(demands_at_t)
        return all_demands

    def _create_environment(self):
        """Initialize/reset environment internal state."""
        self.current_time = 0
        self.nodes = [{"passenger_queue": deque(), "taxis_available": 0} for _ in range(self.num_nodes)]
        self.taxis = [{"current_node": pos, "remaining_time": 0, "is_passenger_trip": False}
                      for pos in self.taxi_positions]
        self.graph = self._build_graph()
        self.total_travel_time = 0
        self.total_demands = 0
        self.total_passengers_served = 0
        self.current_unmet_demand = 0
        self._update_taxi_counts()

    def _build_graph(self):
        """Build directed graph for the grid."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size, create_using=nx.DiGraph)
        for u, v in G.edges():
            u_index = u[0] * self.grid_size + u[1]
            v_index = v[0] * self.grid_size + v[1]
            edge_key = f"{u_index}-{v_index}"
            cost = self.edge_weights.get(edge_key, 1)
            G[u][v]['weight'] = cost
        return nx.convert_node_labels_to_integers(G)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self._create_environment()
        return self._get_state(), {}

    def step(self, actions):
        """Perform a simulation step."""
        if len(actions) != self.num_taxis:
            raise ValueError(f"Expected {self.num_taxis} actions, got {len(actions)}")

        passengers_served_this_step = 0
        travel_time_this_step = 0

        # Add demand for the current timestep
        if self.current_time < len(self.all_demands):
            for origin_dest, count in self.all_demands[self.current_time].items():
                origin, destination = map(int, origin_dest.split('-'))
                for _ in range(count):
                    self.nodes[origin]["passenger_queue"].append(destination)
                self.total_demands += count

        # Reset taxi counts before moves
        for nd in self.nodes:
            nd["taxis_available"] = 0

        # Process taxis
        for i, taxi in enumerate(self.taxis):
            if taxi["remaining_time"] > 0:
                taxi["remaining_time"] -= 1
                if taxi["remaining_time"] == 0 and taxi["is_passenger_trip"]:
                    passengers_served_this_step += 1
                continue

            current_node = taxi["current_node"]
            action = actions[i]

            if self.nodes[current_node]["passenger_queue"]:
                destination = self.nodes[current_node]["passenger_queue"].popleft()
                cost = self._get_path_cost(current_node, destination)
                travel_time_this_step += cost
                taxi["current_node"] = destination
                taxi["remaining_time"] = cost
                taxi["is_passenger_trip"] = True
            else:
                next_node = self._apply_direction(current_node, action)
                if next_node != current_node:
                    cost = self._get_path_cost(current_node, next_node)
                    travel_time_this_step += cost
                    taxi["current_node"] = next_node
                    taxi["remaining_time"] = cost
                    taxi["is_passenger_trip"] = False

        self.total_travel_time += travel_time_this_step
        self.total_passengers_served += passengers_served_this_step
        self._update_taxi_counts()
        self.current_unmet_demand = sum(len(nd["passenger_queue"]) for nd in self.nodes)

        reward = (
            passengers_served_this_step * 0.2
            - 0.05 * self.current_unmet_demand
            - (travel_time_this_step / 100.0)
        )

        self.current_time += 1
        done = (self.current_time >= self.timesteps)

        info = {
            "trips_completed": self.total_passengers_served,
            "total_travel_time": self.total_travel_time,
            "unserved_demand": self.current_unmet_demand,
            "is_last": done
        }

        return self._get_state(), float(reward), done, False, info

    def _apply_direction(self, current_node, direction):
        if direction == 0:
            return current_node
        x, y = divmod(current_node, self.grid_size)
        if direction == 1 and x > 0:
            return (x - 1) * self.grid_size + y
        elif direction == 2 and x < self.grid_size - 1:
            return (x + 1) * self.grid_size + y
        elif direction == 3 and y > 0:
            return x * self.grid_size + (y - 1)
        elif direction == 4 and y < self.grid_size - 1:
            return x * self.grid_size + (y + 1)
        return current_node

    def _get_path_cost(self, node1, node2):
        return nx.shortest_path_length(self.graph, source=node1, target=node2, weight='weight')

    def _update_taxi_counts(self):
        for nd in self.nodes:
            nd["taxis_available"] = 0
        for taxi in self.taxis:
            node_idx = taxi["current_node"]
            self.nodes[node_idx]["taxis_available"] += 1

    def _get_future_demand_forecast(self):
        forecast = np.zeros((self.forecast_horizon, self.num_nodes))
        for i in range(self.forecast_horizon):
            future_time = self.current_time + i
            if future_time < len(self.all_demands):
                for origin_dest, count in self.all_demands[future_time].items():
                    origin, _ = map(int, origin_dest.split('-'))
                    forecast[i, origin] += count
        return forecast.flatten()

    def _get_state(self):
        state = []
        for nd in self.nodes:
            state.extend([len(nd["passenger_queue"]), nd["taxis_available"]])
        future_demand = self._get_future_demand_forecast()
        return np.concatenate([np.array(state, dtype=np.float32), future_demand])
