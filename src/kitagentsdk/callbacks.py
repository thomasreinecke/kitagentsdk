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
    Also reports high-level training state (Rollout vs Optimization).
    """
    def __init__(self, offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None
        self.offset = offset
        self.current_cycle = 0
        self.total_cycles = 0

    def _init_agent(self):
        if not self.agent and self.training_env:
            # Unwrap the environment to find the one holding the kit_client
            # SB3 wraps envs in DummyVecEnv -> Monitor -> YourEnv
            env = self.training_env.envs[0].unwrapped
            if hasattr(env, 'kit_client'):
                self.agent = env.kit_client.agent

    def _on_training_start(self):
        self._init_agent()
        # Calculate total cycles: total_timesteps / n_steps (buffer size)
        if hasattr(self.model, 'n_steps') and self.model.n_steps > 0:
            self.total_cycles = self.locals['total_timesteps'] // self.model.n_steps
        else:
            self.total_cycles = 0

    def _on_rollout_start(self):
        self._init_agent()
        self.current_cycle += 1
        if self.agent and self.total_cycles > 0:
            # Emit event to update UI header
            msg = f"Training running, Rollout (cycle {self.current_cycle}/{self.total_cycles})"
            self.agent.emit_event(msg, "info")

    def _on_rollout_end(self):
        self._init_agent()
        if self.agent and self.total_cycles > 0:
            # Emit event to update UI header before optimization starts
            msg = f"Training running, Optimization (cycle {self.current_cycle}/{self.total_cycles})"
            self.agent.emit_event(msg, "info")

    def _on_step(self) -> bool:
        self._init_agent()

        # Update the environment with the current training progress (Global view)
        env = self.training_env.envs[0].unwrapped
        env.set_training_progress(self.num_timesteps, self.locals['total_timesteps'])
        
        # Report progress back to the kit platform (Relative view)
        # We subtract the offset so the backend sees steps 0..N for this specific stage.
        relative_step = max(0, self.num_timesteps - self.offset)
        if self.agent:
            self.agent.report_progress(relative_step)
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