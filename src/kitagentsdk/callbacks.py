# src/kitagentsdk/callbacks.py
import os
import sys
import time
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class InterimSaveCallback(BaseCallback):
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
    Handles progress reporting, state logging, and graceful stopping.
    Includes console-based progress reporting every 1%.
    """
    def __init__(self, offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None
        self.offset = offset
        self.current_cycle = 0
        self.total_cycles = 0
        self._stop_signal_file = Path("STOP_REQUESTED")
        self._pause_signal_file = Path("PAUSE_REQUESTED")
        
        # Progress tracking
        self._last_logged_pct = -1

    def _init_agent(self):
        if not self.agent and self.training_env:
            env = self.training_env.envs[0].unwrapped
            if hasattr(env, 'kit_client'):
                self.agent = env.kit_client.agent

    def _on_training_start(self):
        self._init_agent()
        if hasattr(self.model, 'n_steps') and self.model.n_steps > 0:
            self.total_cycles = self.locals['total_timesteps'] // self.model.n_steps
        else:
            self.total_cycles = 0

    def _on_rollout_start(self):
        self._init_agent()
        self.current_cycle += 1
        if self.agent and self.total_cycles > 0:
            msg = f"Training running, Rollout (cycle {self.current_cycle}/{self.total_cycles})"
            self.agent.emit_event(msg, "info")

    def _on_rollout_end(self):
        self._init_agent()
        if self.agent and self.total_cycles > 0:
            msg = f"Training running, Optimization (cycle {self.current_cycle}/{self.total_cycles})"
            self.agent.emit_event(msg, "info")

    def _on_step(self) -> bool:
        self._init_agent()

        # Update environment progress (Global)
        env = self.training_env.envs[0].unwrapped
        env.set_training_progress(self.num_timesteps, self.locals['total_timesteps'])
        
        # API Progress Reporting (Relative to this stage)
        relative_step = max(0, self.num_timesteps - self.offset)
        if self.agent:
            self.agent.report_progress(relative_step)

        # Console Progress Logging (Every 1%)
        total_ts = self.locals['total_timesteps']
        if total_ts > 0:
            # We use global num_timesteps here because total_timesteps is usually the global goal
            pct = int((self.num_timesteps / total_ts) * 100)
            if pct > self._last_logged_pct:
                print(f"--- [SDK] Training Progress: {pct}% ({self.num_timesteps}/{total_ts}) ---", file=sys.stderr)
                self._last_logged_pct = pct

        # --- Pause Logic ---
        if self._pause_signal_file.exists():
            if self.agent:
                self.agent.log(f"⏸️ Pause requested at step {self.num_timesteps}. Holding execution...")
                self.agent.emit_event("TRAINING_PAUSED", "warning")
            
            while self._pause_signal_file.exists():
                time.sleep(1)
            
            if self.agent:
                self.agent.log(f"▶️ Resume requested. Continuing training...")
                self.agent.emit_event("TRAINING_RESUMED", "info")

        # --- Graceful Stop Logic ---
        if self._stop_signal_file.exists():
            n_steps = getattr(self.model, 'n_steps', 2048)
            if self.num_timesteps % n_steps == 0:
                if self.agent:
                    self.agent.log(f"🛑 Graceful stop requested. Stopping training at step {self.num_timesteps}.")
                    self.agent.emit_event("TRAINING_STOPPED_GRACEFULLY", "warning")
                return False 

        return True

class SB3MetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.agent = None

    def _on_step(self) -> bool:
        return True 

    def _on_rollout_end(self) -> None:
        if not self.agent:
            env = self.training_env.envs[0].unwrapped
            self.agent = env.kit_client.agent

        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            self.agent.record_metric("rollout/ep_rew_mean", self.num_timesteps, float(ep_rew_mean))
            self.agent.record_metric("rollout/ep_len_mean", self.num_timesteps, float(ep_len_mean))

        if self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith("train/") or key.startswith("time/"):
                    self.agent.record_metric(key, self.num_timesteps, float(value))