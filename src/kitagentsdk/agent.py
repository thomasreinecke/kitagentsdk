# src/kitagentsdk/agent.py
import json
import os
from pathlib import Path
from abc import ABC, abstractmethod
from .kit import KitClient
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

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

        self.kit = KitClient()

    def log(self, message: str):
        """Logs a message to the standard output, ensuring it's captured by kitexec."""
        print(message, flush=True)

    def report_progress(self, step: int):
        """Reports the current training step to a dedicated progress file."""
        progress_file = self.output_path / "progress.log"
        with open(progress_file, "w") as f:
            f.write(str(step))

    def record_metric(self, name: str, step: int, value: float):
        """Records a key-step-value metric to the standard metrics log file."""
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

        This method encapsulates the boilerplate logic for:
        - Setting up standard file paths for artifacts.
        - Wiring up mandatory platform callbacks (logging, interim saves).
        - Executing the model's learn loop.
        - Saving the final model artifact.
        - Saving normalization stats ONLY for newly created models.
        - Cleaning up temporary files.
        """
        final_model_path = self.output_path / "model.zip"
        temp_model_path = self.output_path / "model_temp.zip"
        norm_stats_path = self.output_path / "norm_stats.json"

        # Dynamically import callbacks here to avoid making SB3 a hard dependency for the SDK itself
        from stable_baselines3.common.callbacks import CallbackList
        from .callbacks import InterimSaveCallback, KitLogCallback

        checkpoint_freq = self.config.get("checkpoint_freq", 10000)
        
        # Mandatory callbacks for platform integration
        callbacks = [
            InterimSaveCallback(save_path=str(temp_model_path), save_freq=checkpoint_freq),
            KitLogCallback(),
        ]
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        try:
            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=is_new_model,
                tb_log_name="swing_agent_run",
                callback=CallbackList(callbacks),
            )

            model.save(final_model_path)
            self.log(f"✅ Training complete. Final model saved to {final_model_path}")

            if is_new_model:
                with open(norm_stats_path, 'w') as f:
                    json.dump(env.envs[0].unwrapped.get_norm_stats(), f, indent=4)
                self.log(f"Saved normalization stats to {norm_stats_path}")
            
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        except KeyboardInterrupt:
            self.log("--- 🛑 Training interrupted by user. ---")
            # The calling script should handle sys.exit
        except Exception as e:
            self.log(f"--- ❌ An unexpected error occurred during training: {e} ---")
            raise e

    @abstractmethod
    def train(self):
        """The main training logic for the agent."""
        pass

    def test(self):
        """The main testing/backtesting logic for the agent."""
        self.log("Test command not implemented for this agent.")