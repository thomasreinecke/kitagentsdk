# Kit Agent SDK

[![PyPI Version](https://img.shields.io/pypi/v/kitagentsdk.svg)](https://pypi.org/project/kitagentsdk/)
[![Build Status](https://img.shields.io/travis/com/your-org/kitagentsdk.svg)](https://travis-ci.com/your-org/kitagentsdk)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The `kitagentsdk` is a Python library that provides a standardized, high-level framework for developing agents for the Kit platform. It is designed to abstract away the boilerplate of platform integration, allowing developers to focus entirely on their agent's core logic.

This package includes both the SDK (the `kitagentsdk` library) and a command-line interface (`kitagentcli`) to accelerate development.

### Key Features

*   **Rapid Scaffolding:** Use the `kitagentcli` to generate a complete, working agent project in a single command.
*   **Standardized Agent Structure:** The `BaseAgent` class provides a clear and consistent structure for all agents.
*   **Simplified Platform Interaction:** Helper methods like `self.log()` and `self.record_metric()` handle communication with the Kit platform automatically.
*   **Local Testing:** Agents developed with the SDK are fully runnable on a local machine, enabling rapid testing and debugging.

## Installation

The SDK and CLI are distributed as a single package on PyPI.

```bash
pip install kitagentsdk
```

## Getting Started: Creating Your First Agent

The fastest way to get started is by using the `kitagentcli`.

#### 1. Create a New Agent Project

Run the `new` command to scaffold a new project directory.

```bash
kitagentcli new my-first-agent
```

This will create a new folder named `my-first-agent` with the following structure:

```
my-first-agent/
├── .gitignore
├── README.md
├── manifest.json
├── requirements.txt
└── main.py
```

#### 2. Implement Your Agent Logic

Open `main.py`. The generated file provides a clear starting point for your agent's training logic.

```python
# my-first-agent/main.py
from kitagentsdk import BaseAgent, run_agent
import time

class MyFirstAgent(BaseAgent):
    def train(self):
        """
        Main training logic for the MyFirstAgent.
        """
        self.log(f"--- MyFirstAgent Training Run Initializing ---")
        
        sleep_duration = self.config.get("sleep_duration", 10)
        self.log(f"Configuration loaded. Simulating work for {sleep_duration} seconds.")

        # Simulate training loop
        for i in range(sleep_duration):
            self.log(f"Step {i+1}/{sleep_duration}: Processing data...")
            pnl = (i + 1 - (sleep_duration / 2)) * 0.1 # Example metric
            self.record_metric("pnl", pnl)
            time.sleep(1)

        # Simulate creating an artifact
        artifact_file_path = self.output_path / "model.txt"
        with open(artifact_file_path, "w") as f:
            f.write("This is a trained model.")
        self.log(f"Dummy artifact created at: {artifact_file_path}")

        self.log("--- MyFirstAgent Training Run Finished ---")

if __name__ == "__main__":
    run_agent(MyFirstAgent)
```

#### 3. Run Your Agent Locally

You can test your agent's `train` command directly on your local machine. Create dummy files for the configuration and output path.

```bash
# Create a dummy config file
echo '{"sleep_duration": 5}' > dummy_config.json

# Create a directory for output
mkdir ./local_output

# Run the agent
python main.py train --config dummy_config.json --output-path ./local_output
```

You will see your agent's log messages printed to the console, and the `local_output` directory will contain the generated `model.txt` artifact and `metrics.log` file.

## The `BaseAgent` API

All agents must inherit from `kitagentsdk.BaseAgent`. This base class provides the following helpers:

*   `self.config`: A dictionary containing the configuration parameters for the current run, as defined in `manifest.json` and customized at runtime.
*   `self.output_path`: A `pathlib.Path` object pointing to the directory where all artifacts, models, and metrics should be saved.
*   `self.log(message: str)`: Prints a message to standard output with flushing enabled, ensuring it is captured by the Kit platform's log streamer.
*   `self.record_metric(name: str, value: float)`: Appends a metric to the `metrics.log` file in the output directory in the standard `name,value` format.

## The `manifest.json` File

This file provides the Kit platform with metadata about your agent, including its name, description, and configurable parameters. The CLI generates a default manifest for you to customize.

```json
{
  "name": "my-first-agent",
  "description": "A brief description of what this agent does.",
  "parameters": {
    "sleep_duration": {
      "type": "integer",
      "description": "How long the agent should 'work' in seconds.",
      "default": 10
    }
  }
}
```

## Roadmap

The SDK is under active development. Key features planned for upcoming releases include:
*   A built-in Kit API client (`self.kit`) for standardized data acquisition (e.g., `self.kit.get_candles(...)`).
*   Schema validation for `manifest.json` parameters.

## Contributing

Contributions are welcome! If you have a feature request or bug report, please open an issue on the GitHub repository. For pull requests, please fork the repository and submit your changes to the `main` branch.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```