# episode_callback.py

from stable_baselines3.common.callbacks import BaseCallback

class CustomTensorboardCallback(BaseCallback):
    """
    Logs per-episode metrics to TensorBoard:
      - Summation of episode reward
      - Environment info: 'unserved_demand', 'trips_completed', 'total_travel_time'

    We assume your environment sets 'is_last': True and fills 'infos[i]'
    with final metrics when an episode ends.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_training_start(self) -> None:
        # (Optional) Reset counters at the start of training
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called every environment step. We accumulate rewards,
        and when 'done[i]' is True, we log the final episode metrics.
        """
        # 1) Accumulate this step's rewards in self.episode_reward
        # If using a single env, there's typically 1 reward in self.locals["rewards"]
        # But if using VecEnv with multiple envs, loop over them:
        for i, reward in enumerate(self.locals["rewards"]):
            self.episode_reward += reward

        # 2) Check if any environment in the VecEnv is done
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # The environment i just finished an episode
                # Let's log the final metrics from info
                info = self.locals["infos"][i]
                # Typically, your environment sets "is_last": True on final step,
                # but we can rely on 'done' here.

                # Log the total episode reward
                self.logger.record("episode/reward", self.episode_reward)

                # If your environment places final stats in info
                # e.g. info["unserved_demand"], info["trips_completed"], info["total_travel_time"]
                if "unserved_demand" in info:
                    self.logger.record("episode/unserved_demand", info["unserved_demand"])
                if "trips_completed" in info:
                    self.logger.record("episode/trips_completed", info["trips_completed"])
                if "total_travel_time" in info:
                    self.logger.record("episode/total_travel_time", info["total_travel_time"])

                # Dump to TensorBoard
                self.logger.dump(self.model.num_timesteps)

                # Reset
                self.episode_count += 1
                self.episode_reward = 0.0

        return True  # Return True => training continues
