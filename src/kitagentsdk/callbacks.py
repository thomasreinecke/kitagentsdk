# src/kitagentsdk/callbacks.py
import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class InterimSaveCallback(BaseCallback):
    """Saves the model to a temporary file at a given frequency."""
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
        return True

class KitLogCallback(BaseCallback):
    """
    Handles basic progress reporting via the KitAgentSDK.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None

    def _on_step(self) -> bool:
        if not self.agent:
            env = self.training_env.envs[0].unwrapped
            self.agent = env.kit_client.agent

        # Always update the environment with the current training progress from SB3.
        env = self.training_env.envs[0].unwrapped
        env.set_training_progress(self.num_timesteps, self.locals['total_timesteps'])
        
        # Report progress back to the kit platform
        self.agent.report_progress(self.num_timesteps)
        return True

class SB3MetricsCallback(BaseCallback):
    """
    Streams standard Stable Baselines 3 metrics directly to the backend,
    bypassing the need for TensorBoard log file parsing.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None

    def _on_step(self) -> bool:
        return True # Metrics are logged on rollout end

    def _on_rollout_end(self) -> None:
        if not self.agent:
            env = self.training_env.envs[0].unwrapped
            self.agent = env.kit_client.agent

        # Standard rollout metrics
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            self.agent.record_metric("rollout/ep_rew_mean", self.num_timesteps, float(ep_rew_mean))
            self.agent.record_metric("rollout/ep_len_mean", self.num_timesteps, float(ep_len_mean))

        # Standard training metrics from the last logger update
        if self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith("train/") or key.startswith("time/"):
                    self.agent.record_metric(key, self.num_timesteps, float(value))