# src/kitagentsdk/agent.py
import json
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from .kit import KitClient
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from .context import ContextClient

class BaseAgent(ABC):
    """Abstract base class for all Kit agents."""

    def __init__(self, config_path: str, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = {}

        self.context_client = None
        self.is_test_run = False # Flag to indicate if running in test/simulation mode
        context_path = os.getenv("KIT_CONTEXT_PATH")
        if context_path:
            self.context_client = ContextClient(context_path)
        
        self.kit = KitClient()
        self.kit.agent = self  # Give KitClient a reference back to the agent for event emitting

        if self.context_client:
            self.emit_event("SDK_INITIALIZED")

    def log(self, message: str):
        """Logs a message, sending it to the context server if available, otherwise printing."""
        if self.context_client:
            # Ensure the message has a newline for proper concatenation in the DB.
            log_message = message if message.endswith('\n') else message + '\n'
            self.context_client.log(log_message)
        else:
            # print() adds its own newline, so we send the original message.
            print(message, flush=True)

    def emit_event(self, event_name: str, status: str = "info"):
        """Emits a lifecycle event to the context server if available."""
        if self.context_client:
            self.context_client.emit_event(event_name, status)
        elif not self.is_test_run:
            # Fallback for local debugging, but suppress during test runs
            print(f"[EVENT] {event_name} ({status})", flush=True)

    def report_progress(self, step: int):
        """Reports the current training step to a dedicated progress file."""
        progress_file = self.output_path / "progress.log"
        with open(progress_file, "w") as f:
            f.write(str(step))

    def record_metric(self, name: str, step: int, value: float):
        """Records a key-step-value metric, sending it to the context server."""
        if self.context_client:
            payload = {
                "type": "metric",
                "name": name,
                "step": step,
                "value": value
            }
            self.context_client.send_message(payload)
        else:
            # Fallback for local testing without an executor
            metrics_file = self.output_path / "metrics.log"
            with open(metrics_file, "a") as f:
                f.write(f"{step},{name},{value}\n")

    def orchestrate_sb3_training(
        self,
        env: VecEnv,
        model: BaseAlgorithm,
        is_new_model: bool,
        total_timesteps: int,
        custom_callbacks: list = None,
    ):
        """
        Handles the complete, standardized training lifecycle for a Stable Baselines 3 model.
        """
        final_model_path = self.output_path / "model.zip"
        temp_model_path = self.output_path / "model_temp.zip"
        norm_stats_path = self.output_path / "norm_stats.json"

        from stable_baselines3.common.callbacks import CallbackList
        from .callbacks import InterimSaveCallback, KitLogCallback, SB3MetricsCallback

        checkpoint_freq = self.config.get("checkpoint_freq", 10000)
        
        callbacks = [
            InterimSaveCallback(save_path=str(temp_model_path), save_freq=checkpoint_freq),
            KitLogCallback(),
            SB3MetricsCallback(),
        ]
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        try:
            self.emit_event("TRAINING_LOOP_STARTED")
            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=is_new_model,
                tb_log_name="swing_agent_run",
                callback=CallbackList(callbacks),
            )
            self.emit_event("TRAINING_LOOP_COMPLETED", "success")

            self.emit_event("MODEL_SAVING_STARTED")
            model.save(final_model_path)
            self.emit_event("MODEL_SAVED", "success")

            if is_new_model:
                with open(norm_stats_path, 'w') as f:
                    json.dump(env.envs[0].unwrapped.get_norm_stats(), f, indent=4)
                self.log(f"Saved normalization stats to {norm_stats_path}")
            
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        except KeyboardInterrupt:
            self.log("--- 🛑 Training interrupted by user. ---")
            sys.exit(0) # Graceful exit on Ctrl+C
        except Exception as e:
            self.log(f"--- ❌ An unexpected error occurred during training: {e} ---")
            self.emit_event("AGENT_TRAINING_FAILED", "failure")
            raise e

    @abstractmethod
    def train(self):
        """The main training logic for the agent."""
        pass

    @abstractmethod
    def test(self):
        """The main testing/backtesting logic for the agent."""
        pass