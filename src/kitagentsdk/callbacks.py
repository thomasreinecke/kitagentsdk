# src/kitagentsdk/callbacks.py
import os
from stable_baselines3.common.callbacks import BaseCallback

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
    Handles all logging and metric recording via the KitAgentSDK.
    It logs custom performance metrics at the end of each simulated day,
    using the true training timestep as the x-axis.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped
        if not self.agent:
            self.agent = env.kit_client.agent
        
        # Always update the environment with the current progress
        env.set_training_progress(self.num_timesteps, self.locals['total_timesteps'])
        
        # Report progress back to the kit platform
        self.agent.report_progress(self.num_timesteps)

        # Check for our custom "day_ended" signal from the environment
        info = self.locals["infos"][0]
        if info.get("day_ended"):
            # The agent itself is responsible for logging daily summaries
            # The agent can also record custom metrics as needed
            pass

        return True