# Kit Agent SDK

The `kitagentsdk` is a Python library that provides a standardized, high-level framework for developing agents for the Kit platform. It is designed to abstract away the boilerplate of platform integration, allowing developers to focus entirely on their agent's core logic.

This package includes both the SDK (the `kitagentsdk` library) and a command-line interface (`kitagentcli`) to accelerate development.

### Key Features

*   **Rapid Scaffolding:** Use the `kitagentcli` to generate a complete, working agent project in a single command.
*   **Standardized Agent Structure:** The `BaseAgent` class provides a clear and consistent structure for all agents.
*   **Simplified Platform Interaction:** A rich set of helper methods for logging, metrics, progress reporting, and data fetching that automatically communicate with the Kit platform.
*   **Local Testing:** Agents are fully runnable on a local machine, enabling rapid testing and debugging.

## Installation

The SDK and CLI are distributed as a single package on PyPI.

```bash
pip install kitagentsdk
```

## Getting Started: Creating a New Agent

The fastest way to get started is by using the `kitagentcli`.

#### 1. Create a New Agent Project

Run the `new` command to scaffold a new project directory.

```bash
kitagentcli new my-trading-agent
```

This creates a new folder named `my-trading-agent` with a clean, working structure:

```my-trading-agent/
├── .gitignore
├── README.md
├── manifest.json
├── requirements.txt
└── main.py
```

#### 2. Implement Your Agent's Logic

Open `main.py`. The generated file provides a clear starting point for your agent's training logic, demonstrating the key SDK features.

#### 3. Run Your Agent Locally

You can test your agent's `train` command directly on your local machine.

```bash
# Create a dummy config file (to simulate what the platform provides)
echo '{"total_steps": 100, "learning_rate": 0.01}' > config.json

# Create a directory for output
mkdir ./_output

# Run the agent's training script
python main.py train --config config.json --output-path ./_output
```

You will see your agent's log messages printed to the console, and the `_output` directory will contain the generated `model.zip` artifact and `metrics.log` file.

---

## The `BaseAgent` API

All agents must inherit from `kitagentsdk.BaseAgent`. This base class provides the following helpers:

### Core Properties

*   `self.config`: A dictionary containing the configuration parameters for the current run, as defined in `manifest.json` and customized at runtime in the UI.
*   `self.output_path`: A `pathlib.Path` object pointing to the directory where all artifacts, models, and metrics **must** be saved (e.g., `./_output` locally, or a temporary path when run by `kitexec`).

### UI and Platform Integration Methods

These methods are the primary way your agent communicates its state back to the Kit UI.

*   `self.log(message: str)`:
    Prints a message to the agent's execution log, which is streamed in real-time to the training detail page in the UI. Use this for all status updates and important events.

*   `self.record_metric(name: str, step: int, value: float)`:
    Records a time-series data point. These are automatically collected and visualized as charts in the UI.
    *   **Pro Tip:** Use a `group/name` convention (e.g., `performance/sharpe_ratio`) to automatically group related charts under a common heading in the UI.

*   `self.report_progress(step: int)`:
    Reports the agent's current step count. This updates the progress bars for the run and the overall training in the UI. **It is highly recommended to call this frequently** (e.g., on every step) via a callback.

### The Kit API Client: `self.kit`

The `self.kit` object is an instance of `KitClient` that handles authenticated communication with the Kit backend API.

*   `self.kit.get_training_data(params: dict) -> dict | None`:
    Requests a fully prepared, feature-enriched, and bias-corrected training dataset. This is the standard way for agents to receive market data.
    ```python
    # Example usage inside your environment
    params = {
        "symbol": "YM",
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "features": self.config.get("features", []) # From manifest.json
    }
    data_json = self.kit.get_training_data(params)
    # data_json will contain 'short', 'mid', and 'long' timescale dataframes
    ```

*   `self.kit.download_artifact(artifact_id: UUID, destination_path: str) -> bool`:
    Downloads an artifact from a previous run. This is essential for multi-stage training, allowing a fine-tuning stage to download the `model.zip` or `norm_stats.json` from a pre-training stage. The required artifact IDs are automatically injected into `self.config` by the backend.

---

## Standardized Artifacts

To enable automatic stage transitions, the platform expects two standardized artifact types:
*   `model.zip`: The primary output of any training stage (e.g., the saved SB3 model).
*   `norm_stats.json`: A JSON file containing feature normalization statistics, typically only generated during a pre-training stage.

Your agent should save its final outputs using these filenames in `self.output_path`. The `kitexec` worker will automatically assign the correct type when uploading them.

---

## The `manifest.json` File

This file provides the Kit platform with metadata about your agent.

*   `name`, `description`: Basic info displayed in the UI.
*   `parameters`: Defines the hyperparameters and environment settings that will be exposed to the user in the "Create New Training" wizard.
*   `default_features`: A list of features the agent requires. These are used by `self.kit.get_training_data`.
*   `default_training`: Defines the default multi-stage training plan, including the configuration for each stage.

---

## The `kitagentcli`

A command-line tool to accelerate development.

*   `kitagentcli new <project_name>`:
    Scaffolds a complete, ready-to-run agent project in a new directory named `<project_name>`.

