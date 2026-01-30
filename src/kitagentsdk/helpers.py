# src/kitagentsdk/helpers.py
import argparse
import os
import sys
from .agent import BaseAgent

def run_agent(agent_class: type[BaseAgent]):
    """
    Parses standard command-line arguments and runs the specified agent.
    Ensures final flush of data.
    """
    parser = argparse.ArgumentParser(description=f"Run the {agent_class.__name__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run a training job.")
    train_parser.add_argument("--config", required=False, help="Path to configuration JSON file.")
    train_parser.add_argument("--output-path", required=True, help="Directory for artifacts and logs.")
    train_parser.add_argument("--run-id", required=False, help="Explicit Run ID (overrides env var)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run a backtesting/simulation job.")
    test_parser.add_argument("--config", required=True, help="Path to configuration JSON file.")
    test_parser.add_argument("--output-path", required=True, help="Directory for artifacts and logs.")

    args = parser.parse_args()

    # Override env var if CLI argument provided
    if hasattr(args, "run_id") and args.run_id:
        os.environ["KIT_RUN_ID"] = args.run_id

    agent_instance = agent_class(config_path=args.config, output_path=args.output_path)

    try:
        if args.command == "train":
            agent_instance.train()
        elif args.command == "test":
            agent_instance.is_test_run = True
            agent_instance.test()
    except Exception as e:
        print(f"--- [Helper] Agent execution failed: {e} ---", file=sys.stderr)
        raise e
    finally:
        # Ensure any buffered trades are sent before exit
        if hasattr(agent_instance, 'flush_trades'):
            print("--- [Helper] Ensuring final trade flush... ---", file=sys.stderr)
            agent_instance.flush_trades()