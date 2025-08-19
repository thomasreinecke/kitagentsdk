# src/kitagentsdk/agent.py
import json
import os
from pathlib import Path
from abc import ABC, abstractmethod
from .kit import KitClient

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

    def record_metric(self, name: str, step: int, value: float):
        """Records a key-step-value metric to the standard metrics log file."""
        metrics_file = self.output_path / "metrics.log"
        with open(metrics_file, "a") as f:
            f.write(f"{step},{name},{value}\n")

    @abstractmethod
    def train(self):
        """The main training logic for the agent."""
        pass

    def test(self):
        """The main testing/backtesting logic for the agent."""
        self.log("Test command not implemented for this agent.")